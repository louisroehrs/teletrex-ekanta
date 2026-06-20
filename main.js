const { app, BrowserWindow, session, ipcMain } = require('electron');
const path     = require('path');
const http     = require('http');
const crypto   = require('crypto');
const { execFile } = require('child_process');

// Enable WebGPU with Metal backend on macOS
app.setName("TeleTrex Ekanta")

app.setAboutPanelOptions({
  applicationName: 'TeleTrex Ekanta',
  applicationVersion: '1.0.0',
  copyright: '© 2026 Louis Roehrs',
  credits: 'TeleTrex',
  website: 'https://teletrex.com',
});

// Electron (unlike Chrome) does not expose navigator.gpu by default — these two
// switches turn WebGPU on. WebGPUDeveloperFeatures unlocks the subgroup support
// onnxruntime-web's FlashAttention shader needs.
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch('enable-features', 'WebGPU,WebGPUDeveloperFeatures');
// NOTE: deliberately NOT enabling WebGPUExperimentalFeatures or
// `enable-dawn-features=allow_unsafe_apis` — those route Dawn through
// experimental kernels suspected of corrupting Gemma 4 generation.
app.commandLine.appendSwitch('enable-gpu-rasterization');

const SERVER_PORT = 42069;

// ---------------------------------------------------------------------------
// Shared state shared between HTTP server and IPC handlers
// ---------------------------------------------------------------------------
let mainWindow = null;

// modelState is updated by the renderer via IPC whenever a model loads/unloads
let modelState = { loaded: false, modelId: null, modelName: null };

// pending HTTP requests waiting on renderer inference
// Map<requestId, { res, isStream, buffer, created }>
const pending = new Map();

// ---------------------------------------------------------------------------
// IPC: renderer pushes model status updates
// ---------------------------------------------------------------------------
ipcMain.on('server:model-status', (_event, state) => {
  modelState = state;
});

// IPC: renderer sends a streaming chunk back to HTTP client
ipcMain.on('server:inference-chunk', (_event, { requestId, content }) => {
  const req = pending.get(requestId);
  if (!req) return;

  if (req.isStream) {
    if (req.format === 'anthropic') {
      writeAnthropicEvent(req.res, 'content_block_delta', {
        type: 'content_block_delta', index: 0,
        delta: { type: 'text_delta', text: content },
      });
    } else {
      req.res.write(`data: ${JSON.stringify({
        id: requestId, object: 'chat.completion.chunk', created: req.created,
        model: modelState.modelId ?? 'unknown',
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      })}\n\n`);
    }
  } else {
    req.buffer += content;
  }
});

// IPC: renderer signals stream is complete
ipcMain.on('server:inference-done', (_event, { requestId, promptTokens, completionTokens }) => {
  const req = pending.get(requestId);
  if (!req) return;
  pending.delete(requestId);

  const inTok  = promptTokens     ?? 0;
  const outTok = completionTokens ?? 0;
  const model  = modelState.modelId ?? 'unknown';

  if (req.format === 'anthropic') {
    if (req.isStream) {
      writeAnthropicEvent(req.res, 'content_block_stop',  { type: 'content_block_stop', index: 0 });
      writeAnthropicEvent(req.res, 'message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: outTok },
      });
      writeAnthropicEvent(req.res, 'message_stop', { type: 'message_stop' });
      req.res.end();
    } else {
      sendJson(req.res, 200, {
        id: requestId, type: 'message', role: 'assistant', model,
        content: [{ type: 'text', text: req.buffer }],
        stop_reason: 'end_turn', stop_sequence: null,
        usage: { input_tokens: inTok, output_tokens: outTok },
      });
    }
  } else {
    if (req.isStream) {
      req.res.write(`data: ${JSON.stringify({
        id: requestId, object: 'chat.completion.chunk', created: req.created, model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      })}\n\n`);
      req.res.write('data: [DONE]\n\n');
      req.res.end();
    } else {
      sendJson(req.res, 200, {
        id: requestId, object: 'chat.completion', created: req.created, model,
        choices: [{ index: 0, message: { role: 'assistant', content: req.buffer }, finish_reason: 'stop' }],
        usage: { prompt_tokens: inTok, completion_tokens: outTok, total_tokens: inTok + outTok },
      });
    }
  }
});

// IPC: renderer requests GPU info via system_profiler (authoritative VRAM source on macOS)
ipcMain.handle('gpu:get-info', () => new Promise((resolve) => {
  execFile('system_profiler', ['SPDisplaysDataType', '-json'], { timeout: 5000 }, (err, stdout) => {
    if (err) { resolve(null); return; }
    try { resolve(JSON.parse(stdout).SPDisplaysDataType ?? null); }
    catch { resolve(null); }
  });
}));

// IPC: renderer signals an error
ipcMain.on('server:inference-error', (_event, { requestId, error }) => {
  const req = pending.get(requestId);
  if (!req) return;
  pending.delete(requestId);
  if (req.isStream) {
    if (req.format === 'anthropic') writeAnthropicEvent(req.res, 'error', { type: 'error', error: { type: 'server_error', message: error } });
    else req.res.write(`data: ${JSON.stringify({ error })}\n\n`);
    req.res.end();
  } else {
    sendJson(req.res, 500, { error: { message: error, type: 'server_error' } });
  }
});

function writeAnthropicEvent(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

// ---------------------------------------------------------------------------
// HTTP server — OpenAI-compatible endpoints
// ---------------------------------------------------------------------------
function createHttpServer() {
  const server = http.createServer((req, res) => {
    // CORS — allow any local or tool to hit the API
    res.setHeader('Access-Control-Allow-Origin',  '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    const { pathname } = new URL(req.url, `http://localhost:${SERVER_PORT}`);

    if (req.method === 'GET' && pathname === '/v1/models') {
      return handleModels(res);
    }

    if (req.method === 'POST' && pathname === '/v1/chat/completions') {
      return handleChatCompletions(req, res);
    }

    if (req.method === 'POST' && pathname === '/v1/messages') {
      return handleMessages(req, res);
    }

    if (pathname === '/') {
      return sendJson(res, 200, { name: 'Ekanta', version: '1.0.0', status: 'ok' });
    }

    sendJson(res, 404, { error: { message: 'Not found', type: 'invalid_request_error' } });
  });

  server.listen(SERVER_PORT, '127.0.0.1', () => {
    console.log(`Ekanta API listening on http://127.0.0.1:${SERVER_PORT}`);
  });

  // Renderer pulls the port once it's ready
  ipcMain.handle('server:get-port', () => SERVER_PORT);

  server.on('error', (err) => {
    console.error('HTTP server error:', err.message);
  });

  return server;
}

function handleModels(res) {
  const data = modelState.loaded
    ? [{
        id:       modelState.modelId,
        object:   'model',
        created:  Math.floor(Date.now() / 1000),
        owned_by: 'ekanta',
      }]
    : [];
  sendJson(res, 200, { object: 'list', data });
}

function handleChatCompletions(req, res) {
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    let payload;
    try { payload = JSON.parse(body); }
    catch {
      return sendJson(res, 400, { error: { message: 'Invalid JSON', type: 'invalid_request_error' } });
    }

    if (!modelState.loaded) {
      return sendJson(res, 503, { error: { message: 'No model loaded. Load a model in the Ekanta UI first.', type: 'server_error' } });
    }

    const messages   = payload.messages   ?? [];
    const isStream   = payload.stream     ?? false;
    const temperature = payload.temperature ?? 0.7;
    const maxTokens  = payload.max_tokens  ?? 2048;

    if (!messages.length) {
      return sendJson(res, 400, { error: { message: '`messages` is required', type: 'invalid_request_error' } });
    }

    const requestId = 'chatcmpl-' + crypto.randomBytes(8).toString('hex');
    const created   = Math.floor(Date.now() / 1000);

    if (isStream) {
      res.writeHead(200, {
        'Content-Type':  'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection':    'keep-alive',
      });
      // Send role delta first (OpenAI convention)
      const roleDelta = {
        id: requestId, object: 'chat.completion.chunk', created,
        model: modelState.modelId ?? 'unknown',
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      };
      res.write(`data: ${JSON.stringify(roleDelta)}\n\n`);
    }

    pending.set(requestId, { res, isStream, buffer: '', created, format: 'openai' });

    // Forward to renderer
    if (mainWindow) {
      mainWindow.webContents.send('server:inference-request', {
        requestId, messages, temperature, maxTokens,
      });
    } else {
      pending.delete(requestId);
      if (isStream) { res.write('data: [DONE]\n\n'); res.end(); }
      else sendJson(res, 503, { error: { message: 'App window not ready', type: 'server_error' } });
    }
  });
}

function handleMessages(req, res) {
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    let payload;
    try { payload = JSON.parse(body); }
    catch { return sendJson(res, 400, { error: { message: 'Invalid JSON', type: 'invalid_request_error' } }); }

    if (!modelState.loaded) {
      return sendJson(res, 503, { error: { message: 'No model loaded. Load a model in the Ekanta UI first.', type: 'server_error' } });
    }

    // Convert Anthropic format → internal (OpenAI-style messages array)
    const messages    = [...(payload.messages ?? [])];
    const isStream    = payload.stream      ?? false;
    const temperature = payload.temperature ?? 0.7;
    const maxTokens   = payload.max_tokens  ?? 2048;

    if (payload.system) messages.unshift({ role: 'system', content: payload.system });
    if (!messages.length) return sendJson(res, 400, { error: { message: '`messages` is required', type: 'invalid_request_error' } });

    const requestId = 'msg_' + crypto.randomBytes(12).toString('hex');
    const created   = Math.floor(Date.now() / 1000);
    const model     = modelState.modelId ?? 'unknown';

    if (isStream) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
      writeAnthropicEvent(res, 'message_start', {
        type: 'message_start',
        message: { id: requestId, type: 'message', role: 'assistant', content: [], model, stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 1 } },
      });
      writeAnthropicEvent(res, 'content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
      writeAnthropicEvent(res, 'ping', { type: 'ping' });
    }

    pending.set(requestId, { res, isStream, buffer: '', created, format: 'anthropic' });

    if (mainWindow) {
      mainWindow.webContents.send('server:inference-request', { requestId, messages, temperature, maxTokens });
    } else {
      pending.delete(requestId);
      if (isStream) { writeAnthropicEvent(res, 'message_stop', { type: 'message_stop' }); res.end(); }
      else sendJson(res, 503, { error: { message: 'App window not ready', type: 'server_error' } });
    }
  });
}

function sendJson(res, status, body) {
  const json = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(json);
}

// ---------------------------------------------------------------------------
// Electron lifecycle
// ---------------------------------------------------------------------------
function createWindow() {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Cross-Origin-Opener-Policy':   ['same-origin'],
        'Cross-Origin-Embedder-Policy': ['require-corp'],
      },
    });
  });

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0f0f13',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      allowRunningInsecureContent: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'app', 'index.html'));

  mainWindow.on('closed', () => { mainWindow = null; });

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(() => {
  createHttpServer();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
