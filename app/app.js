import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

// ---------------------------------------------------------------------------
// Model catalogue — all fit within 8 GB VRAM (Q4F16 quantization)
// ---------------------------------------------------------------------------
const MODELS = [


  { id: 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC',      name: 'DeepSeek-R1 · 7B',      params: '7B',   vram: 4.5, tag: 'Reasoning', tagColor: 'tag-purple', desc: 'Chain-of-thought reasoning distilled' },
  { id: 'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC',     name: 'DeepSeek-R1 · 8B',      params: '8B',   vram: 5.0, tag: 'Reasoning', tagColor: 'tag-purple', desc: 'Strongest local reasoning model under 8 GB' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC',                    name: 'Gemma 2 · 2B',          params: '2B',   vram: 1.5, tag: 'Google',    tagColor: 'tag-green',  desc: 'Google — efficient instruction-tuned' },
  { id: 'gemma-2-9b-it-q4f16_1-MLC',                    name: 'Gemma 2 · 9B',           params: '9B',   vram: 5.5, tag: 'Google',    tagColor: 'tag-green',  desc: 'Google — top quality at 9B parameters' },
//  { id: 'gemma3-1b-it-q4f16_1-MLC',                      name: 'Gemma 3 · 1B',           params: '1B',   vram: 0.7, tag: 'Google',    tagColor: 'tag-green',  desc: 'Google — ultra-light Gemma 3, runs anywhere',
//    overrides: { context_window_size: 4096, sliding_window_size: -1 } },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',            name: 'Llama 3.2 · 3B',        params: '3B',   vram: 2.0, tag: 'Meta',      tagColor: 'tag-orange', desc: 'Meta — fast general-purpose assistant' },
  { id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC',            name: 'Llama 3.1 · 8B',        params: '8B',   vram: 5.0, tag: 'Meta',      tagColor: 'tag-orange', desc: 'Meta flagship — balanced quality & speed' },
  { id: 'Mistral-7B-Instruct-v0.3-q4f16_1-MLC',         name: 'Mistral 7B v0.3',        params: '7B',   vram: 4.5, tag: '★ Popular', tagColor: 'tag-gold',   desc: 'Mistral AI — excellent 7B benchmark' },
  { id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',            name: 'Phi 3.5 Mini',           params: '3.8B', vram: 2.5, tag: 'Microsoft', tagColor: 'tag-blue',   desc: 'Microsoft — punches above its weight' },
  { id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',            name: 'Qwen 2.5 · 0.5B',     params: '0.5B', vram: 0.4, tag: 'Tiny',      tagColor: 'tag-blue',   desc: 'Alibaba — ultra-light, instant replies' },
  { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',              name: 'Qwen 2.5 · 3B',         params: '3B',   vram: 2.0, tag: 'Alibaba',   tagColor: 'tag-green',  desc: 'Strong multilingual & coding support' },
  { id: 'Qwen2.5-7B-Instruct-q4f16_1-MLC',              name: 'Qwen 2.5 · 7B',         params: '7B',   vram: 4.5, tag: 'Coding',    tagColor: 'tag-green',  desc: 'Best-in-class coding & reasoning at 7B' },
  { id: 'Qwen3-8B-q4f16_1-MLC',                         name: 'Qwen 3 · 8B',         params: '8B',   vram: 5.3 , tag: 'General',    tagColor: 'tag-green',  desc: 'Best-in-class reasoning at 8B' },
  { id: 'Qwen3.5-9B-q4f16_1-MLC',                       name: 'Qwen 3.5 · 9B',         params: '9B',   vram: 6.4 , tag: 'General',    tagColor: 'tag-green',  desc: 'Bigger and slower than Qwen 3' },
  { id: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',            name: 'SmolLM2 · 1.7B',       params: '1.7B', vram: 1.1, tag: 'Fast',      tagColor: 'tag-blue',   desc: 'HuggingFace — surprisingly capable small model' },
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
let serverGenerating = false;

const cachedModelIds = new Set();

async function refreshCachedModels() {
  if (typeof webllm.hasModelInCache !== 'function') return;
  await Promise.all(MODELS.map(async (m) => {
    try {
      const cached = await webllm.hasModelInCache(m.id);
      cached ? cachedModelIds.add(m.id) : cachedModelIds.delete(m.id);
    } catch {}
  }));
  MODELS.forEach(m => updateModelCardStatus(m.id));
}

function updateModelCardStatus(modelId) {
  const card = modelListEl?.querySelector(`[data-id="${CSS.escape(modelId)}"]`);
  if (!card) return;
  let el = card.querySelector('.model-status');
  const isActive = modelId === loadedModelId;
  const isCached = cachedModelIds.has(modelId);
  if (!isActive && !isCached) { el?.remove(); return; }
  if (!el) {
    el = document.createElement('div');
    card.appendChild(el);
  }
  el.className = `model-status${isActive ? ' model-status--active' : ' model-status--cached'}`;
  el.innerHTML = isActive
    ? '<span class="status-dot"></span>Active'
    : '<span class="status-dot"></span>Cached';
}

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
const btnCopyConv        = document.getElementById('btnCopyConv');
const btnCopyConvLabel   = document.getElementById('btnCopyConvLabel');
const btnToggleConv      = document.getElementById('btnToggleConv');
const btnToggleSidebar   = document.getElementById('btnToggleSidebar');
const convPanelEl        = document.querySelector('.conv-panel');
const sidebarEl          = document.querySelector('.sidebar');

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
    const prev = loadedModelId;
    engine = null;
    loadedModelId = null;
    window.electronAPI?.sendModelStatus({ loaded: false, modelId: null, modelName: null });
    if (prev) updateModelCardStatus(prev);
  }

  try {
    const engineConfig = {
      initProgressCallback: (report) => {
        const pct = Math.round((report.progress || 0) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = report.text
          ? report.text.replace(/\[.*?\]\s*/, '').slice(0, 60)
          : `${pct}%`;
      },
    };
    if (selectedModel.overrides) {
      engineConfig.appConfig = {
        ...webllm.prebuiltAppConfig,
        model_list: webllm.prebuiltAppConfig.model_list.map(m =>
          m.model_id === selectedModel.id
            ? { ...m, overrides: { ...(m.overrides ?? {}), ...selectedModel.overrides } }
            : m
        ),
      };
    }
    engine = await webllm.CreateMLCEngine(selectedModel.id, engineConfig);

    const prevModelId = loadedModelId;
    loadedModelId = selectedModel.id;
    cachedModelIds.add(selectedModel.id);
    if (prevModelId) updateModelCardStatus(prevModelId);
    updateModelCardStatus(loadedModelId);
    window.electronAPI?.sendModelStatus({ loaded: true, modelId: selectedModel.id, modelName: selectedModel.name });
    progressFill.style.width = '100%';
    progressText.textContent = 'Ready';

    inputBox.disabled = false;
    btnSend.disabled = false;
    btnClear.disabled = false;
    btnCopyConv.disabled = false;
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
        Select a model from the sidebar and click <strong>Load Model</strong> to begin. Once a model is cached, it is available to use offline. Load and cache the desired models, turn off wifi, and everything still runs.
        Everything runs locally and privately on your Mac, inflight, no cloud, no AI service, no network needed.
      </p>
      <div class="welcome-features">
        <div class="feature"><span class="feature-icon">🔒</span><span>100% local &amp; private</span></div>
        <div class="feature"><span class="feature-icon">⚡</span><span>Runs fast on your machine's GPU.</span></div>
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
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm,  '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,   '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,    '<h1>$1</h1>');
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  html = html.replace(/^---$/gm, '<hr>');

  // Tables — must run before paragraph wrapping
  html = html.replace(
    /^(\|.+)\r?\n(\|[\s|:|-]+)\r?\n((?:\|.+\r?\n?)*)/gm,
    (_, headerLine, sepLine, bodyLines) => {
      const cells = line => line.split('|').slice(1, -1).map(c => c.trim());
      const aligns = cells(sepLine).map(c =>
        /^:-+:$/.test(c) ? 'center' : /^-+:$/.test(c) ? 'right' : 'left'
      );
      const th = cells(headerLine)
        .map((c, i) => `<th style="text-align:${aligns[i]}">${c}</th>`).join('');
      const trs = bodyLines.trim().split('\n').filter(Boolean)
        .map(row => `<tr>${cells(row).map((c, i) =>
          `<td style="text-align:${aligns[i]}">${c}</td>`).join('')}</tr>`).join('');
      return `<table class="md-table"><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
    }
  );

  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;
  html = html.replace(/<p>\s*(<(?:pre|ul|table|h[1-4]|hr))/g, '$1');
  html = html.replace(/(<\/(?:pre|ul|table|h[1-4])>|<hr>)\s*<\/p>/g, '$1');

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
// Copy conversation as rich text (HTML + plain text fallback)
// ---------------------------------------------------------------------------
async function copyConversationAsMarkdown() {
  const conv = getConv(activeConvId);
  if (!conv || !conv.messages.length) return;

  const modelLabel = conv.modelName ?? 'Ekanta';
  const date = new Date(conv.createdAt).toLocaleDateString(undefined, { dateStyle: 'medium' });

  // ── Rich HTML ────────────────────────────────────────────────────────────
  const msgHtml = conv.messages.map(msg => {
    const isUser = msg.role === 'user';
    const speaker = isUser ? 'You' : modelLabel;
    const speakerColor = isUser ? '#7c6af7' : '#0f766e';
    const rendered = renderMarkdown(msg.content.trim());
    return `
      <div style="margin:0 0 18px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.6;color:#1c1917;">
        <div style="font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:${speakerColor};margin-bottom:4px;">${speaker}</div>
        <div style="padding-left:2px;">${rendered}</div>
      </div>`;
  }).join('');

  const html = `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:720px;padding:24px;">
      <h1 style="margin:0 0 4px;font-size:20px;font-weight:700;color:#1c1917;">${escapeHtml(conv.title)}</h1>
      <p style="margin:0 0 24px;font-size:12px;color:#78716c;">${escapeHtml(modelLabel)} · ${escapeHtml(date)}</p>
      <hr style="border:none;border-top:1px solid #e7e5e4;margin:0 0 24px;">
      ${msgHtml}
    </div>`;

  // ── Plain text fallback ──────────────────────────────────────────────────
  const plain = [conv.title, `${modelLabel} · ${date}`, '']
    .concat(conv.messages.flatMap(msg => {
      const speaker = msg.role === 'user' ? 'You' : modelLabel;
      return [`${speaker}:`, msg.content.trim(), ''];
    }))
    .join('\n');

  await navigator.clipboard.write([
    new ClipboardItem({
      'text/html':  new Blob([html],  { type: 'text/html' }),
      'text/plain': new Blob([plain], { type: 'text/plain' }),
    }),
  ]);

  btnCopyConvLabel.textContent = 'Copied!';
  setTimeout(() => { btnCopyConvLabel.textContent = 'Copy'; }, 2000);
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

// Panel toggle buttons
function applyPanelState() {
  const convCollapsed    = localStorage.getItem('ekanta_conv_collapsed')    === '1';
  const sidebarCollapsed = localStorage.getItem('ekanta_sidebar_collapsed') === '1';
  convPanelEl.classList.toggle('collapsed', convCollapsed);
  sidebarEl.classList.toggle('collapsed', sidebarCollapsed);
  btnToggleConv.classList.toggle('active', !convCollapsed);
  btnToggleSidebar.classList.toggle('active', !sidebarCollapsed);
}

btnToggleConv.addEventListener('click', () => {
  const next = localStorage.getItem('ekanta_conv_collapsed') !== '1' ? '1' : '0';
  localStorage.setItem('ekanta_conv_collapsed', next);
  applyPanelState();
});

btnToggleSidebar.addEventListener('click', () => {
  const next = localStorage.getItem('ekanta_sidebar_collapsed') !== '1' ? '1' : '0';
  localStorage.setItem('ekanta_sidebar_collapsed', next);
  applyPanelState();
});

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

btnCopyConv.addEventListener('click', copyConversationAsMarkdown);

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
// Neural network background animation
// ---------------------------------------------------------------------------
function initNeuralBackground() {
  const canvas = document.getElementById('neuralBg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const COLORS = [
    { r: 124, g: 106, b: 247 }, // accent purple
    { r: 124, g: 106, b: 247 },
    { r: 124, g: 106, b: 247 },
    { r: 45,  g: 212, b: 191 }, // teal accent
    { r: 167, g: 139, b: 250 }, // lighter purple
  ];

  let nodes = [], signals = [];

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    buildNodes();
  }

  function buildNodes() {
    const area = canvas.width * canvas.height;
    const count = Math.max(22, Math.min(55, Math.floor(area / 14000)));
    nodes = Array.from({ length: count }, () => {
      const c = COLORS[Math.floor(Math.random() * COLORS.length)];
      return {
        x:          Math.random() * canvas.width,
        y:          Math.random() * canvas.height,
        vx:         (Math.random() - 0.5) * 0.28,
        vy:         (Math.random() - 0.5) * 0.28,
        phase:      Math.random() * Math.PI * 2,
        phaseSpeed: 0.004 + Math.random() * 0.006,
        waveAmp:    0.18 + Math.random() * 0.22,
        r:          1.2 + Math.random() * 1.8,
        depth:      0.3 + Math.random() * 0.7, // 0=far, 1=near
        c,
      };
    });
    signals = [];
  }

  function spawnSignal() {
    if (nodes.length < 2) return;
    const threshold = 170;
    // pick a random edge that exists
    for (let attempt = 0; attempt < 20; attempt++) {
      const i = Math.floor(Math.random() * nodes.length);
      const j = Math.floor(Math.random() * nodes.length);
      if (i === j) continue;
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      if (Math.sqrt(dx * dx + dy * dy) < threshold) {
        signals.push({ from: i, to: j, t: 0, speed: 0.008 + Math.random() * 0.012 });
        break;
      }
    }
  }

  let lastSignalTime = 0;

  function draw(ts) {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const threshold = 170;

    // Update nodes
    nodes.forEach(n => {
      n.phase += n.phaseSpeed;
      n.x += n.vx;
      n.y += n.vy + Math.sin(n.phase) * n.waveAmp;
      if (n.x < -60) n.x = W + 60;
      if (n.x > W + 60) n.x = -60;
      if (n.y < -60) n.y = H + 60;
      if (n.y > H + 60) n.y = -60;
    });

    // Spawn signals periodically
    if (ts - lastSignalTime > 800 + Math.random() * 1200) {
      spawnSignal();
      lastSignalTime = ts;
    }

    // Draw edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < threshold) {
          const fade   = 1 - dist / threshold;
          const depth  = (nodes[i].depth + nodes[j].depth) * 0.5;
          const alpha  = fade * fade * depth * 0.18;
          const { r, g, b } = nodes[i].c;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = depth * 0.8;
          ctx.stroke();
        }
      }
    }

    // Draw signals
    signals = signals.filter(sig => {
      sig.t += sig.speed;
      if (sig.t > 1) return false;
      const a = nodes[sig.from], b = nodes[sig.to];
      const sx = a.x + (b.x - a.x) * sig.t;
      const sy = a.y + (b.y - a.y) * sig.t;
      const grd = ctx.createRadialGradient(sx, sy, 0, sx, sy, 6);
      grd.addColorStop(0,   'rgba(200,180,255,0.9)');
      grd.addColorStop(0.4, 'rgba(124,106,247,0.4)');
      grd.addColorStop(1,   'rgba(124,106,247,0)');
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
      return true;
    });

    // Draw nodes
    nodes.forEach(n => {
      const pulse = 0.85 + Math.sin(n.phase) * 0.15;
      const r     = n.r * pulse * n.depth;
      const alpha = 0.25 + n.depth * 0.45;
      const { r: cr, g: cg, b: cb } = n.c;

      // Glow halo
      const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 5);
      grd.addColorStop(0,   `rgba(${cr},${cg},${cb},${alpha * 0.3})`);
      grd.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`);
      ctx.beginPath();
      ctx.arc(n.x, n.y, r * 5, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Node core
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  new ResizeObserver(resize).observe(canvas.parentElement);
  resize();
  requestAnimationFrame(draw);
}

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
  applyPanelState();
  initNeuralBackground();

  // Pre-select first model
  selectModel(MODELS[0]);

  const ok = await detectGPU();

  // Check which models are already cached (runs in background)
  refreshCachedModels();
  if (!ok) btnLoad.title = 'WebGPU not available';

  // Wire up server IPC
  if (window.electronAPI) {
    window.electronAPI.getServerPort().then((port) => updateServerBadge(port));
    window.electronAPI.onServerInferenceRequest((data) => handleServerInference(data));
  }
})();
