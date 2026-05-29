import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

// ---------------------------------------------------------------------------
// Model catalogue — all fit within 8 GB VRAM (Q4F16 quantization)
// ---------------------------------------------------------------------------
const MODELS = [
  { id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',            name: 'Qwen 2.5 · 0.5B',     params: '0.5B', vram: 0.4, tag: 'Tiny',      tagColor: 'tag-blue',   desc: 'Alibaba — ultra-light, instant replies' },
  { id: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',            name: 'SmolLM2 · 1.7B',       params: '1.7B', vram: 1.1, tag: 'Fast',      tagColor: 'tag-blue',   desc: 'HuggingFace — surprisingly capable small model' },
  { id: 'DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC',    name: 'DeepSeek-R1 · 1.5B',   params: '1.5B', vram: 1.0, tag: 'Reasoning', tagColor: 'tag-purple', desc: 'DeepSeek — distilled reasoning model' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC',                    name: 'Gemma 2 · 2B',          params: '2B',   vram: 1.5, tag: 'Google',    tagColor: 'tag-green',  desc: 'Google — efficient instruction-tuned' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',            name: 'Llama 3.2 · 3B',        params: '3B',   vram: 2.0, tag: 'Meta',      tagColor: 'tag-orange', desc: 'Meta — fast general-purpose assistant' },
  { id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',            name: 'Phi 3.5 Mini',           params: '3.8B', vram: 2.5, tag: 'Microsoft', tagColor: 'tag-blue',   desc: 'Microsoft — punches above its weight' },
  { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',              name: 'Qwen 2.5 · 3B',         params: '3B',   vram: 2.0, tag: 'Alibaba',   tagColor: 'tag-green',  desc: 'Strong multilingual & coding support' },
  { id: 'Mistral-7B-Instruct-v0.3-q4f16_1-MLC',         name: 'Mistral 7B v0.3',        params: '7B',   vram: 4.5, tag: '★ Popular', tagColor: 'tag-gold',   desc: 'Mistral AI — excellent 7B benchmark' },
  { id: 'Qwen2.5-7B-Instruct-q4f16_1-MLC',              name: 'Qwen 2.5 · 7B',         params: '7B',   vram: 4.5, tag: 'Coding',    tagColor: 'tag-green',  desc: 'Best-in-class coding & reasoning at 7B' },
  { id: 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC',      name: 'DeepSeek-R1 · 7B',      params: '7B',   vram: 4.5, tag: 'Reasoning', tagColor: 'tag-purple', desc: 'Chain-of-thought reasoning distilled' },
  { id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC',            name: 'Llama 3.1 · 8B',        params: '8B',   vram: 5.0, tag: 'Meta',      tagColor: 'tag-orange', desc: 'Meta flagship — balanced quality & speed' },
  { id: 'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC',     name: 'DeepSeek-R1 · 8B',      params: '8B',   vram: 5.0, tag: 'Reasoning', tagColor: 'tag-purple', desc: 'Strongest local reasoning model under 8 GB' },
  { id: 'gemma-2-9b-it-q4f16_1-MLC',                    name: 'Gemma 2 · 9B',           params: '9B',   vram: 5.5, tag: 'Google',    tagColor: 'tag-green',  desc: 'Google — top quality at 9B parameters' },
];

// ---------------------------------------------------------------------------
// Storage keys
// ---------------------------------------------------------------------------
const STORAGE_CONVS     = 'ekanta_conversations';
const STORAGE_SETTINGS  = 'ekanta_settings';
const STORAGE_ACTIVE    = 'ekanta_active_conv';

// ---------------------------------------------------------------------------
// Settings (persisted to localStorage)
// ---------------------------------------------------------------------------
const DEFAULT_SETTINGS = { systemPrompt: '', temperature: 0.7, maxTokens: 2048 };
let settings = { ...DEFAULT_SETTINGS, ...(JSON.parse(localStorage.getItem(STORAGE_SETTINGS) || 'null') ?? {}) };

function saveSettings() {
  localStorage.setItem(STORAGE_SETTINGS, JSON.stringify(settings));
}

// ---------------------------------------------------------------------------
// Conversations (persisted to localStorage)
// ---------------------------------------------------------------------------
function loadConversations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_CONVS) || '[]'); } catch { return []; }
}

function persistConversations() {
  localStorage.setItem(STORAGE_CONVS, JSON.stringify(conversations));
}

let conversations = loadConversations();
let activeConvId  = localStorage.getItem(STORAGE_ACTIVE) || null;

function getConv(id) { return conversations.find(c => c.id === id) ?? null; }

function createConversation() {
  const conv = {
    id: `conv_${Date.now()}`,
    title: 'New Conversation',
    modelId: null,
    modelName: null,
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  conversations.unshift(conv);
  persistConversations();
  return conv;
}

function deleteConversation(id) {
  conversations = conversations.filter(c => c.id !== id);
  persistConversations();
  if (activeConvId === id) {
    activeConvId = conversations[0]?.id ?? null;
    localStorage.setItem(STORAGE_ACTIVE, activeConvId ?? '');
  }
}

function setActiveConversation(id) {
  activeConvId = id;
  localStorage.setItem(STORAGE_ACTIVE, id ?? '');
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------
let engine           = null;
let loadedModelId    = null;
let selectedModel    = null;
let isGenerating     = false;
let serverGenerating = false;  // true while handling an HTTP server request

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const convListEl     = document.getElementById('convList');
const btnNewChat     = document.getElementById('btnNewChat');
const tabBtnModels   = document.getElementById('tabBtnModels');
const tabBtnSettings = document.getElementById('tabBtnSettings');
const paneModels     = document.getElementById('paneModels');
const paneSettings   = document.getElementById('paneSettings');
const modelListEl    = document.getElementById('modelList');
const btnLoad        = document.getElementById('btnLoad');
const progressWrap   = document.getElementById('progressWrap');
const progressFill   = document.getElementById('progressFill');
const progressText   = document.getElementById('progressText');
const messagesEl     = document.getElementById('messages');
const inputBox       = document.getElementById('inputBox');
const btnSend        = document.getElementById('btnSend');
const btnClear       = document.getElementById('btnClear');
const modelStatus    = document.getElementById('modelStatus');
const gpuLabel       = document.getElementById('gpuLabel');
const gpuBadge       = document.getElementById('gpuBadge');
const systemPromptInput  = document.getElementById('systemPromptInput');
const tempSlider         = document.getElementById('tempSlider');
const tempValue          = document.getElementById('tempValue');
const maxTokensSlider    = document.getElementById('maxTokensSlider');
const maxTokensValue     = document.getElementById('maxTokensValue');
const btnSaveSettings    = document.getElementById('btnSaveSettings');
const settingsSaved      = document.getElementById('settingsSaved');

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------
async function detectGPU() {
  if (!navigator.gpu) {
    gpuLabel.textContent = 'WebGPU not available';
    gpuBadge.classList.add('gpu-error');
    showError('WebGPU is not available. Ensure you are running Electron 28+ with a supported GPU.');
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      gpuLabel.textContent = 'No GPU adapter';
      gpuBadge.classList.add('gpu-error');
      return false;
    }
    const info = await adapter.requestAdapterInfo();
    gpuLabel.textContent = info.description || info.vendor || 'GPU';
    gpuBadge.classList.add('gpu-ready');
    return true;
  } catch {
    gpuLabel.textContent = 'GPU error';
    gpuBadge.classList.add('gpu-error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  const isModels = tab === 'models';
  tabBtnModels.classList.toggle('active', isModels);
  tabBtnSettings.classList.toggle('active', !isModels);
  paneModels.classList.toggle('hidden', !isModels);
  paneSettings.classList.toggle('hidden', isModels);
}

// ---------------------------------------------------------------------------
// Settings UI
// ---------------------------------------------------------------------------
function syncSettingsUI() {
  systemPromptInput.value  = settings.systemPrompt;
  tempSlider.value         = settings.temperature;
  tempValue.textContent    = settings.temperature.toFixed(2);
  maxTokensSlider.value    = settings.maxTokens;
  maxTokensValue.textContent = settings.maxTokens;
}

// ---------------------------------------------------------------------------
// Build model list
// ---------------------------------------------------------------------------
function buildModelList() {
  modelListEl.innerHTML = '';
  MODELS.forEach((m) => {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.id = m.id;
    const vramPct = Math.round((m.vram / 8) * 100);
    card.innerHTML = `
      <div class="model-card-top">
        <div class="model-name">${m.name}</div>
        <span class="model-tag ${m.tagColor}">${m.tag}</span>
      </div>
      <div class="model-desc">${m.desc}</div>
      <div class="model-meta">
        <span class="model-params">${m.params}</span>
        <div class="vram-bar-wrap" title="${m.vram} GB VRAM">
          <div class="vram-bar-track">
            <div class="vram-bar-fill" style="width:${vramPct}%"></div>
          </div>
          <span class="vram-label">${m.vram} GB</span>
        </div>
      </div>
    `;
    card.addEventListener('click', () => selectModel(m));
    modelListEl.appendChild(card);
  });
}

function selectModel(m) {
  if (isGenerating) return;
  selectedModel = m;
  document.querySelectorAll('.model-card').forEach((c) => {
    c.classList.toggle('selected', c.dataset.id === m.id);
  });
  const alreadyLoaded = loadedModelId === m.id;
  btnLoad.disabled = alreadyLoaded;
  btnLoad.textContent = alreadyLoaded ? 'Loaded ✓' : 'Load Model';
}

// ---------------------------------------------------------------------------
// Load model
// ---------------------------------------------------------------------------
async function loadModel() {
  if (!selectedModel || isGenerating) return;

  btnLoad.disabled = true;
  progressWrap.style.display = 'flex';
  progressFill.style.width = '0%';
  progressFill.classList.remove('progress-error');
  progressText.textContent = 'Connecting…';
  modelStatus.textContent = `Loading ${selectedModel.name}…`;

  if (engine) {
    try { await engine.unload(); } catch (_) {}
    engine = null;
    loadedModelId = null;
    window.electronAPI?.sendModelStatus({ loaded: false, modelId: null, modelName: null });
  }

  try {
    engine = await webllm.CreateMLCEngine(selectedModel.id, {
      initProgressCallback: (report) => {
        const pct = Math.round((report.progress || 0) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = report.text
          ? report.text.replace(/\[.*?\]\s*/, '').slice(0, 60)
          : `${pct}%`;
      },
    });

    loadedModelId = selectedModel.id;
    window.electronAPI?.sendModelStatus({ loaded: true, modelId: selectedModel.id, modelName: selectedModel.name });
    progressFill.style.width = '100%';
    progressText.textContent = 'Ready';

    inputBox.disabled = false;
    btnSend.disabled = false;
    btnClear.disabled = false;
    inputBox.placeholder = `Chat with ${selectedModel.name}…`;
    modelStatus.textContent = `${selectedModel.name} · ${selectedModel.params} · ${selectedModel.vram} GB`;
    btnLoad.textContent = 'Loaded ✓';

    // Make sure there's an active conversation
    if (!activeConvId || !getConv(activeConvId)) {
      const conv = createConversation();
      setActiveConversation(conv.id);
      renderConvList();
    }

    const conv = getConv(activeConvId);
    if (conv && conv.messages.length === 0) {
      conv.modelId = selectedModel.id;
      conv.modelName = selectedModel.name;
      persistConversations();
    }

    renderConvList();
    clearMessagesUI();
    loadConversationMessages();
    inputBox.focus();

    setTimeout(() => { progressWrap.style.display = 'none'; }, 2000);
  } catch (err) {
    console.error('Model load error:', err);
    progressText.textContent = 'Failed — check console';
    progressFill.style.width = '0%';
    progressFill.classList.add('progress-error');
    btnLoad.disabled = false;
    btnLoad.textContent = 'Retry';
    modelStatus.textContent = 'Load failed';
    showError(`Failed to load model: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Conversations UI
// ---------------------------------------------------------------------------
function renderConvList() {
  convListEl.innerHTML = '';

  if (conversations.length === 0) {
    const empty = document.createElement('div');
    empty.style.cssText = 'padding:20px 8px; text-align:center; font-size:11px; color:var(--text-muted);';
    empty.textContent = 'No conversations yet';
    convListEl.appendChild(empty);
    return;
  }

  conversations.forEach((conv) => {
    const item = document.createElement('div');
    item.className = `conv-item${conv.id === activeConvId ? ' active' : ''}`;
    item.dataset.id = conv.id;

    const metaParts = [];
    if (conv.modelName) metaParts.push(conv.modelName);
    if (conv.messages.length) metaParts.push(`${conv.messages.length} msg${conv.messages.length !== 1 ? 's' : ''}`);
    const meta = metaParts.join(' · ') || formatDate(conv.createdAt);

    item.innerHTML = `
      <div class="conv-item-body">
        <div class="conv-title">${escapeHtml(conv.title)}</div>
        <div class="conv-meta">${escapeHtml(meta)}</div>
      </div>
      <button class="conv-delete" title="Delete conversation" aria-label="Delete">×</button>
    `;

    item.querySelector('.conv-item-body').addEventListener('click', () => switchConversation(conv.id));
    item.querySelector('.conv-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
      renderConvList();
      if (activeConvId === conv.id || !activeConvId) {
        clearMessagesUI();
        showWelcome();
      }
    });

    convListEl.appendChild(item);
  });
}

function switchConversation(id) {
  if (id === activeConvId) return;
  setActiveConversation(id);
  renderConvList();
  clearMessagesUI();
  loadConversationMessages();
}

function loadConversationMessages() {
  const conv = getConv(activeConvId);
  if (!conv || conv.messages.length === 0) {
    showWelcome();
    return;
  }
  hideWelcome();
  conv.messages.forEach((msg) => appendMessageUI(msg.role, msg.content));
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
async function sendMessage() {
  if (!engine || isGenerating || serverGenerating) return;
  const text = inputBox.value.trim();
  if (!text) return;

  // Ensure active conversation
  if (!activeConvId || !getConv(activeConvId)) {
    const conv = createConversation();
    setActiveConversation(conv.id);
    renderConvList();
  }

  const conv = getConv(activeConvId);
  if (!conv) return;

  isGenerating = true;
  inputBox.value = '';
  resizeInput();
  btnSend.disabled = true;
  inputBox.disabled = true;

  hideWelcome();

  // Auto-title from first user message
  if (conv.messages.filter(m => m.role === 'user').length === 0) {
    conv.title = text.slice(0, 42).trim() + (text.length > 42 ? '…' : '');
    if (!conv.modelId) {
      conv.modelId   = loadedModelId;
      conv.modelName = selectedModel?.name ?? null;
    }
  }

  conv.messages.push({ role: 'user', content: text });
  conv.updatedAt = Date.now();
  persistConversations();
  renderConvList();

  appendMessageUI('user', text);

  // Build messages for the API (inject system prompt if set)
  const apiMessages = [];
  if (settings.systemPrompt.trim()) {
    apiMessages.push({ role: 'system', content: settings.systemPrompt.trim() });
  }
  conv.messages.forEach(m => apiMessages.push(m));

  const assistantEl  = appendMessageUI('assistant', '');
  const contentEl    = assistantEl.querySelector('.msg-content');
  let fullText       = '';

  try {
    const stream = await engine.chat.completions.create({
      messages: apiMessages,
      stream: true,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      fullText += delta;
      contentEl.innerHTML = renderMarkdown(fullText);
      scrollToBottom();
    }

    conv.messages.push({ role: 'assistant', content: fullText });
    conv.updatedAt = Date.now();
    persistConversations();
    renderConvList();
  } catch (err) {
    console.error('Generation error:', err);
    contentEl.innerHTML = `<span class="error-text">Error: ${escapeHtml(err.message)}</span>`;
  } finally {
    isGenerating = false;
    btnSend.disabled = false;
    inputBox.disabled = false;
    inputBox.focus();
  }
}

// ---------------------------------------------------------------------------
// Messages UI
// ---------------------------------------------------------------------------
function appendMessageUI(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message message-${role}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'You' : 'AI'}</div>
    <div class="msg-bubble">
      <div class="msg-content">${text ? renderMarkdown(text) : '<span class="cursor"></span>'}</div>
    </div>
  `;
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function clearMessagesUI() {
  messagesEl.innerHTML = '';
}

function showWelcome() {
  if (!document.getElementById('welcome')) {
    const w = document.createElement('div');
    w.id = 'welcome';
    w.className = 'welcome';
    w.innerHTML = `
      <div class="welcome-icon-wrap">
        <img class="welcome-icon-img" src="../assets/icon.iconset/icon_512x512.png" alt="Ekanta icon" />
      </div>
      <h1 class="welcome-h1">Ekanta</h1>
      <p class="welcome-brand-line">by TeleTrex</p>
      <p class="welcome-desc">
        Select a model from the sidebar and click <strong>Load Model</strong> to begin.<br/>
        Everything runs locally on your Mac via WebGPU — no cloud, no API keys.
      </p>
      <div class="welcome-features">
        <div class="feature"><span class="feature-icon">🔒</span><span>100% local &amp; private</span></div>
        <div class="feature"><span class="feature-icon">⚡</span><span>WebGPU accelerated</span></div>
        <div class="feature"><span class="feature-icon">💾</span><span>Models cached locally</span></div>
      </div>
    `;
    messagesEl.appendChild(w);
  }
}

function hideWelcome() {
  document.getElementById('welcome')?.remove();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showError(msg) {
  const el = document.createElement('div');
  el.className = 'error-banner';
  el.textContent = msg;
  messagesEl.appendChild(el);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Minimal Markdown → HTML
// ---------------------------------------------------------------------------
function renderMarkdown(text) {
  let html = escapeHtml(text);

  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trimEnd()}</code></pre>`
  );

  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;
  html = html.replace(/<p>\s*(<(?:pre|ul|h[1-3]))/g, '$1');
  html = html.replace(/(<\/(?:pre|ul|h[1-3])>)\s*<\/p>/g, '$1');

  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(ts) {
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

// ---------------------------------------------------------------------------
// HTTP server inference (called via IPC from main process)
// ---------------------------------------------------------------------------
async function handleServerInference({ requestId, messages, temperature, maxTokens }) {
  if (!engine) {
    window.electronAPI.sendServerError({ requestId, error: 'No model loaded' });
    return;
  }
  if (isGenerating || serverGenerating) {
    window.electronAPI.sendServerError({ requestId, error: 'Engine busy — try again shortly' });
    return;
  }

  serverGenerating = true;
  btnSend.disabled = true;
  inputBox.disabled = true;

  let promptTokens     = 0;
  let completionTokens = 0;

  try {
    const stream = await engine.chat.completions.create({
      messages,
      stream: true,
      temperature,
      max_tokens: maxTokens,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content ?? '';
      if (content) {
        window.electronAPI.sendServerChunk({ requestId, content });
        completionTokens++;
      }
      if (chunk.usage) {
        promptTokens     = chunk.usage.prompt_tokens     ?? promptTokens;
        completionTokens = chunk.usage.completion_tokens ?? completionTokens;
      }
    }

    window.electronAPI.sendServerDone({ requestId, promptTokens, completionTokens });
  } catch (err) {
    console.error('Server inference error:', err);
    window.electronAPI.sendServerError({ requestId, error: err.message });
  } finally {
    serverGenerating = false;
    if (engine) {
      btnSend.disabled = false;
      inputBox.disabled = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Server status badge in titlebar
// ---------------------------------------------------------------------------
function updateServerBadge(port) {
  const badge = document.getElementById('serverBadge');
  if (!badge) return;
  badge.querySelector('.server-port').textContent = `:${port}`;
  badge.classList.add('server-running');
}

// ---------------------------------------------------------------------------
// Input auto-resize
// ---------------------------------------------------------------------------
function resizeInput() {
  inputBox.style.height = 'auto';
  inputBox.style.height = Math.min(inputBox.scrollHeight, 160) + 'px';
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
tabBtnModels.addEventListener('click',   () => switchTab('models'));
tabBtnSettings.addEventListener('click', () => switchTab('settings'));

btnNewChat.addEventListener('click', () => {
  const conv = createConversation();
  setActiveConversation(conv.id);
  renderConvList();
  clearMessagesUI();
  showWelcome();
  if (engine) {
    inputBox.disabled = false;
    btnSend.disabled = false;
    btnClear.disabled = false;
  }
});

btnLoad.addEventListener('click', loadModel);

btnSend.addEventListener('click', sendMessage);

btnClear.addEventListener('click', () => {
  const conv = getConv(activeConvId);
  if (!conv) return;
  conv.messages = [];
  conv.updatedAt = Date.now();
  persistConversations();
  renderConvList();
  clearMessagesUI();
  showWelcome();
  inputBox.focus();
});

inputBox.addEventListener('input', resizeInput);

inputBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

tempSlider.addEventListener('input', () => {
  tempValue.textContent = parseFloat(tempSlider.value).toFixed(2);
});

maxTokensSlider.addEventListener('input', () => {
  maxTokensValue.textContent = maxTokensSlider.value;
});

btnSaveSettings.addEventListener('click', () => {
  settings.systemPrompt = systemPromptInput.value;
  settings.temperature  = parseFloat(tempSlider.value);
  settings.maxTokens    = parseInt(maxTokensSlider.value, 10);
  saveSettings();
  settingsSaved.style.display = 'block';
  setTimeout(() => { settingsSaved.style.display = 'none'; }, 2000);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async () => {
  buildModelList();
  syncSettingsUI();

  // Restore active conversation or start fresh
  if (activeConvId && !getConv(activeConvId)) {
    activeConvId = conversations[0]?.id ?? null;
  }

  renderConvList();
  showWelcome();

  // Pre-select first model
  selectModel(MODELS[0]);

  const ok = await detectGPU();
  if (!ok) btnLoad.title = 'WebGPU not available';

  // Wire up server IPC
  if (window.electronAPI) {
    window.electronAPI.onServerStarted(({ port }) => updateServerBadge(port));
    window.electronAPI.onServerInferenceRequest((data) => handleServerInference(data));
  }
})();
