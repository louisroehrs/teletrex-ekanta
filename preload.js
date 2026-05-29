const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  arch:     process.arch,
  versions: {
    electron: process.versions.electron,
    chrome:   process.versions.chrome,
    node:     process.versions.node,
  },

  // ── Server bridge: renderer → main ────────────────────────────────────
  sendModelStatus:    (data) => ipcRenderer.send('server:model-status',    data),
  sendServerChunk:    (data) => ipcRenderer.send('server:inference-chunk', data),
  sendServerDone:     (data) => ipcRenderer.send('server:inference-done',  data),
  sendServerError:    (data) => ipcRenderer.send('server:inference-error', data),

  // ── Server bridge: main → renderer ────────────────────────────────────
  getServerPort:            ()   => ipcRenderer.invoke('server:get-port'),
  onServerInferenceRequest: (cb) => ipcRenderer.on('server:inference-request', (_e, d) => cb(d)),
});
