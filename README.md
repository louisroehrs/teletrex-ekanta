# Ekanta by TeleTrex

**Private, local AI chat for Mac. No cloud. No subscription. No data ever leaves your machine.**

[Learn more at teletrex.com/products/ekanta](https://teletrex.com/products/ekanta)

---

## What is Ekanta?

Ekanta runs powerful AI language models directly on your Mac's GPU, even ones on Intel-based Macs. Every question you ask and every answer you receive stays entirely on your device — nothing is sent to any server, ever. There is no account, no API key, and no monthly fee.

---

## System Requirements

- macOS 12 Monterey or later
- Internet connection on first use (to download models to your machine — after that, works fully offline)

**Performance note:** Ekanta runs on any supported Mac, but runs noticeably faster on:
- **Intel Macs with a dedicated AMD GPU** (e.g. MacBook Pro 16" 2019–2020 with Radeon Pro)
- **Apple Silicon Macs** (M1 and later) — unified memory gives these machines excellent inference speed

---

## Download

Go to the [Releases page on GitHub](https://github.com/teletrex/ekanta/releases) and download the version that matches your Mac:

| Your Mac | File to download |
|---|---|
| Apple Silicon (M1, M2, M3, M4…) | `Ekanta-x.x.x-arm64.dmg` |
| Intel Mac | `Ekanta-x.x.x.dmg` |

Not sure which you have? Click the **Apple menu → About This Mac**. If it says "Apple M…" you have Apple Silicon. If it says "Intel Core" you have an Intel Mac.  On Intel Macs, Ekanta will look for the high-performace GPU which has its own dedicated 8Gig of VRAM in addition to the RAM attached to the CPU.

---

## Install

1. Download the .dmg file from Releases for your Mac.
2. Open the downloaded `.dmg` file.
3. A window opens showing the Ekanta icon and an Applications folder.
4. Drag the **TeleTrex Ekanta** icon onto the **Applications** folder.
5. Eject the disk image (drag it to Trash or press ⌘E).

> **First launch:** macOS may show a security prompt because Ekanta is not yet notarized. If that happens, go to **System Settings → Privacy & Security**, scroll down, and click **Open Anyway**.

---

## Getting Started

### 1. Open Ekanta
Find Ekanta in your Applications folder and double-click it.

### 2. Select a model
The **Models** panel is on the right side of the window. Each model has a name, a size, and a bar showing how much GPU memory it needs.

**Recommended starting model: Qwen 2.5 · 3B**
It's fast, capable, and uses only 2 GB of GPU memory — a great balance of speed and quality for everyday tasks.

### 3. Load the model
Click **Load Model**. Ekanta will download the model on first use (a few hundred MB to a few GB depending on the model). This only happens once — after that the model is cached locally and loads instantly.

The status badge at the top will turn green and show **Active** when the model is ready.

### 4. Start chatting
Type your question in the box at the bottom and press **Return** (or click the send button).

That's it. **Your questions and the model's answers never leave your Mac.** Everything is generated locally by your GPU in real time.

---

## Tips

- **Offline use:** Once a model is downloaded, Ekanta works with no internet connection at all — on a plane, in a café, the desert,  anywhere.
- **Multiple models:** You can download and cache as many models as your disk has space for. Switch between them freely.
- **System prompt:** Open the **Settings** tab to set a system prompt that shapes how the model behaves across all your conversations.
- **Copy conversations:** Use the **Copy** button in the bottom toolbar to copy any conversation as formatted rich text, ready to paste into a document or email.
- **Local API:** Ekanta also runs a local OpenAI-compatible API server on port `42069`, so developer tools and coding assistants can use it as a backend.  Note: some coding UIs look for a Visual Language Model and that is not currently provided.

---

## More Information

[teletrex.com/products/ekanta](https://teletrex.com/products/ekanta)

---

*Ekanta is built by [TeleTrex](https://teletrex.com). Copyright © 2026 TeleTrex and Louis Roehrs.*

Ekanta is based on AI technology and may provide incorrect answers.