// WebLLM loaded from CDN — internet required for first load (models too)
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

// ---------------------------------------------------------------------------
// Model catalogue — all fit within 8 GB VRAM (Q4F16 quantization)
// ---------------------------------------------------------------------------
const MODELS = [
  {
    id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
    name: 'Qwen 2.5 · 0.5B',
    params: '0.5B',
    vram: 0.4,
    tag: 'Tiny',
    tagColor: 'tag-blue',
    desc: 'Alibaba — ultra-light, instant replies',
  },
  {
    id: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
    name: 'SmolLM2 · 1.7B',
    params: '1.7B',
    vram: 1.1,
    tag: 'Fast',
    tagColor: 'tag-blue',
    desc: 'HuggingFace — surprisingly capable small model',
  },
  {
    id: 'DeepSeek-R1-Distill-Qwen-1.5B-q4f16_1-MLC',
    name: 'DeepSeek-R1 · 1.5B',
    params: '1.5B',
    vram: 1.0,
    tag: 'Reasoning',
    tagColor: 'tag-purple',
    desc: 'DeepSeek — distilled reasoning model',
  },
  {
    id: 'gemma-2-2b-it-q4f16_1-MLC',
    name: 'Gemma 2 · 2B',
    params: '2B',
    vram: 1.5,
    tag: 'Google',
    tagColor: 'tag-green',
    desc: 'Google — efficient instruction-tuned',
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.2 · 3B',
    params: '3B',
    vram: 2.0,
    tag: 'Meta',
    tagColor: 'tag-orange',
    desc: 'Meta — fast general-purpose assistant',
  },
  {
    id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
    name: 'Phi 3.5 Mini',
    params: '3.8B',
    vram: 2.5,
    tag: 'Microsoft',
    tagColor: 'tag-blue',
    desc: 'Microsoft — punches above its weight',
  },
  {
    id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    name: 'Qwen 2.5 · 3B',
    params: '3B',
    vram: 2.0,
    tag: 'Alibaba',
    tagColor: 'tag-green',
    desc: 'Strong multilingual & coding support',
  },
  {
    id: 'Mistral-7B-Instruct-v0.3-q4f16_1-MLC',
    name: 'Mistral 7B v0.3',
    params: '7B',
    vram: 4.5,
    tag: '★ Popular',
    tagColor: 'tag-gold',
    desc: 'Mistral AI — excellent 7B benchmark',
  },
  {
    id: 'Qwen2.5-7B-Instruct-q4f16_1-MLC',
    name: 'Qwen 2.5 · 7B',
    params: '7B',
    vram: 4.5,
    tag: 'Coding',
    tagColor: 'tag-green',
    desc: 'Best-in-class coding & reasoning at 7B',
  },
  {
    id: 'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC',
    name: 'DeepSeek-R1 · 7B',
    params: '7B',
    vram: 4.5,
    tag: 'Reasoning',
    tagColor: 'tag-purple',
    desc: 'Chain-of-thought reasoning distilled',
  },
  {
    id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.1 · 8B',
    params: '8B',
    vram: 5.0,
    tag: 'Meta',
    tagColor: 'tag-orange',
    desc: 'Meta flagship — balanced quality & speed',
  },
  {
    id: 'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC',
    name: 'DeepSeek-R1 · 8B',
    params: '8B',
    vram: 5.0,
    tag: 'Reasoning',
    tagColor: 'tag-purple',
    desc: 'Strongest local reasoning model under 8 GB',
  },
  {
    id: 'gemma-2-9b-it-q4f16_1-MLC',
    name: 'Gemma 2 · 9B',
    params: '9B',
    vram: 5.5,
    tag: 'Google',
    tagColor: 'tag-green',
    desc: 'Google — top quality at 9B parameters',
  },
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let engine = null;
let selectedModel = null;
let isGenerating = false;
let chatHistory = [];

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const modelListEl = document.getElementById('modelList');
const btnLoad = document.getElementById('btnLoad');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const messagesEl = document.getElementById('messages');
const welcome = document.getElementById('welcome');
const inputBox = document.getElementById('inputBox');
const btnSend = document.getElementById('btnSend');
const btnClear = document.getElementById('btnClear');
const modelStatus = document.getElementById('modelStatus');
const gpuLabel = document.getElementById('gpuLabel');
const gpuBadge = document.getElementById('gpuBadge');

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------
async function detectGPU() {
  if (!navigator.gpu) {
    gpuLabel.textContent = 'WebGPU not available';
    gpuBadge.classList.add('gpu-error');
    showError(
      'WebGPU is not available in this context. ' +
      'Make sure you are running Electron 28+ and the GPU is supported.'
    );
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      gpuLabel.textContent = 'No GPU adapter';
      gpuBadge.classList.add('gpu-error');
      return false;
    }

    const info = await adapter.requestAdapterInfo();
    const name = info.description || info.vendor || 'GPU';
    gpuLabel.textContent = `${name}`;
    gpuBadge.classList.add('gpu-ready');
    return true;
  } catch (e) {
    gpuLabel.textContent = 'GPU error';
    gpuBadge.classList.add('gpu-error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// Build model list UI
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
  const alreadyLoaded = engine && engine._modelId === m.id;
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
  progressText.textContent = 'Connecting…';
  modelStatus.textContent = `Loading ${selectedModel.name}…`;

  // Unload previous engine
  if (engine) {
    try { await engine.unload(); } catch (_) {}
    engine = null;
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

    // Tag the engine with which model is loaded
    engine._modelId = selectedModel.id;

    progressFill.style.width = '100%';
    progressText.textContent = 'Ready';

    inputBox.disabled = false;
    btnSend.disabled = false;
    btnClear.disabled = false;
    inputBox.placeholder = `Chat with ${selectedModel.name}…`;
    modelStatus.textContent = `${selectedModel.name} · ${selectedModel.params} · ${selectedModel.vram} GB`;

    btnLoad.textContent = 'Loaded ✓';
    chatHistory = [];
    clearMessages();
    inputBox.focus();

    setTimeout(() => {
      progressWrap.style.display = 'none';
    }, 2000);
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
// Chat
// ---------------------------------------------------------------------------
async function sendMessage() {
  if (!engine || isGenerating) return;
  const text = inputBox.value.trim();
  if (!text) return;

  isGenerating = true;
  inputBox.value = '';
  resizeInput();
  btnSend.disabled = true;
  inputBox.disabled = true;

  welcome?.remove();

  appendMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  const assistantEl = appendMessage('assistant', '');
  const contentEl = assistantEl.querySelector('.msg-content');
  let fullText = '';

  try {
    const stream = await engine.chat.completions.create({
      messages: chatHistory,
      stream: true,
      temperature: 0.7,
      max_tokens: 2048,
    });

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      fullText += delta;
      contentEl.innerHTML = renderMarkdown(fullText);
      scrollToBottom();
    }

    chatHistory.push({ role: 'assistant', content: fullText });
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
// Message rendering
// ---------------------------------------------------------------------------
function appendMessage(role, text) {
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

function clearMessages() {
  messagesEl.innerHTML = '';
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
// Simple Markdown → HTML (code blocks, inline code, bold, italic, links)
// ---------------------------------------------------------------------------
function renderMarkdown(text) {
  let html = escapeHtml(text);

  // Fenced code blocks
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code class="lang-${lang}">${code.trimEnd()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Unordered list items
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Line breaks (preserve paragraphs)
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = `<p>${html}</p>`;

  // Clean up empty paragraphs around block elements
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
btnLoad.addEventListener('click', loadModel);

btnSend.addEventListener('click', sendMessage);

btnClear.addEventListener('click', () => {
  chatHistory = [];
  clearMessages();
  inputBox.focus();
});

inputBox.addEventListener('input', resizeInput);

inputBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
(async () => {
  buildModelList();

  const ok = await detectGPU();
  if (!ok) {
    btnLoad.title = 'WebGPU not available';
  }

  // Pre-select the first model
  selectModel(MODELS[0]);
})();
