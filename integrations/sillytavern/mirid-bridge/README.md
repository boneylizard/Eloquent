# Mirid Bridge for SillyTavern

Mirid Bridge lets SillyTavern use a running Mirid desktop installation without a SillyTavern server plugin.

## What works

- OpenAI-compatible streaming chat through Mirid's local GGUF models and configured API endpoints
- Automatic character narration through Mirid TTS
- Microphone transcription into SillyTavern's message box through Mirid STT
- SillyTavern Image Generation through Mirid's local stable-diffusion.cpp engine
- OpenAI-compatible speech, transcription, and image endpoints for other clients

## Install during development

Copy this folder to:

```text
SillyTavern/public/scripts/extensions/third-party/mirid-bridge
```

Restart SillyTavern, open **Extensions**, then expand **Mirid Bridge**.

Once this folder is published as its own Git repository, users can install it from SillyTavern's **Install extension** dialog by pasting that repository URL.

## Connect

1. Start Mirid and leave its address as `http://127.0.0.1:8000` unless Mirid shows a different port.
2. If Mirid remote access has a password, enter it in Mirid Bridge too.
3. Select **Test connection**.
4. Select **Find voices** to load Mirid's installed voice catalogue.

The extension panel contains the current text and image setup instructions for SillyTavern.

## Network safety

Mirid listens on localhost by default. If you deliberately expose it to your network, configure Mirid's remote-access password first and use that password in SillyTavern.
