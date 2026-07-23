# Mirid AI Backend for SillyTavern

[Mirid](https://mirid.ai) is a Windows desktop app for downloading, running and talking to AI models. It can run GGUF models on your computer or connect to hosted AI providers, with built-in tools for voices, speech recognition and image generation.

This extension connects SillyTavern to a running Mirid installation. It does not install Mirid or download models by itself.

## Before you begin

1. [Download and install the current Mirid release](https://github.com/boneylizard/Eloquent/releases/latest).
2. Open Mirid and complete its first-run setup.
3. Install this extension in SillyTavern.

## What works

- OpenAI-compatible streaming chat through Mirid's local GGUF models and configured API endpoints
- Automatic character narration through Mirid TTS
- Microphone transcription into SillyTavern's message box through Mirid STT
- SillyTavern Image Generation through Mirid's local stable-diffusion.cpp engine
- OpenAI-compatible speech, transcription, and image endpoints for other clients

## Install

1. In SillyTavern, open **Extensions** and choose **Install extension**.
2. Paste `https://github.com/boneylizard/mirid-sillytavern-bridge`.
3. Restart SillyTavern, then expand **Mirid AI Backend** under Extensions.

For local development, copy this folder to `SillyTavern/public/scripts/extensions/third-party/mirid-bridge`.

## Connect

1. Close SillyTavern. Mirid and SillyTavern both use port `8000` by default.
2. Open SillyTavern's `config.yaml`, set `port: 8001`, then restart SillyTavern.
3. Start Mirid and leave its address as `http://127.0.0.1:8000` unless Mirid shows a different port.
4. If Mirid remote access has a password, enter it in Mirid Bridge too.
5. Select **Test connection**.
6. Select **Find voices** to load Mirid's installed voice catalogue.

The extension panel contains the current text and image setup instructions for SillyTavern.

## Network safety

Mirid listens on localhost by default. If you deliberately expose it to your network, configure Mirid's remote-access password first and use that password in SillyTavern.
