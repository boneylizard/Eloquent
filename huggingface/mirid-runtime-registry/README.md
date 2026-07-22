---
title: Mirid Runtime Registry
emoji: 🪞
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
short_description: Verified local inference packages for Mirid
models:
  - openai/whisper-large-v3-turbo
  - UsefulSensors/moonshine-streaming-tiny
  - nvidia/parakeet-tdt-0.6b-v2
  - nvidia/parakeet-tdt-0.6b-v3
  - nvidia/stt_zh_conformer_transducer_large
  - nvidia/nemotron-speech-streaming-en-0.6b
  - mudler/parakeet-cpp-gguf
  - ResembleAI/chatterbox
  - ResembleAI/chatterbox-turbo
  - ResembleAI/chatterbox-nano
  - openbmb/VoxCPM2
  - hexgrad/Kokoro-82M
  - LiquidAI/LFM2.5-VL-450M
  - LiquidAI/LFM2.5-VL-450M-Extract
  - LiquidAI/LFM2.5-VL-1.6B
  - LiquidAI/LFM2.5-VL-1.6B-Extract
  - google/gemma-3-4b-it
datasets:
  - boneylizardwizard/mirid-runtime-packs
tags:
  - local-inference
  - automatic-speech-recognition
  - text-to-speech
  - image-text-to-text
  - gguf
---

# Mirid Runtime Registry

This Space presents Mirid's independently updateable local-inference packages. The binary files live in the public [`boneylizardwizard/mirid-runtime-packs`](https://huggingface.co/datasets/boneylizardwizard/mirid-runtime-packs) Dataset repository, where every published file is tied to an upstream release, size and SHA-256 digest.

The packages are redistributions of their named upstream projects. Their original licences and release notes remain authoritative. A listed package is not a promise that its hardware path has been tested by Mirid unless the validation field says so.

## ASR models

- [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
- [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)
- [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [`nvidia/stt_zh_conformer_transducer_large`](https://huggingface.co/nvidia/stt_zh_conformer_transducer_large)
- [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf)

## TTS and voice models

- [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox)
- [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [`ResembleAI/chatterbox-nano`](https://huggingface.co/ResembleAI/chatterbox-nano)
- [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2)
- [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)

## Vision models

- [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
- [`LiquidAI/LFM2.5-VL-450M-Extract`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract)
- [`LiquidAI/LFM2.5-VL-1.6B`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)
- [`LiquidAI/LFM2.5-VL-1.6B-Extract`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract)
- [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)

These are the concrete Hugging Face repositories named by Mirid for automatic speech recognition, text to speech, voice conversion and vision. Some are optional rather than bundled with every runtime installation.
