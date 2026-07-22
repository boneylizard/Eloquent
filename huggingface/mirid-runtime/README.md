---
license: agpl-3.0
tags:
  - mirid
  - llama-cpp
  - gguf
  - automatic-speech-recognition
  - text-to-speech
  - image-text-to-text
  - windows
  - cuda
  - vulkan
---

# Mirid Runtime for Windows

This repository hosts the frozen local-inference runtime used by the Mirid desktop app. It is a release channel, not a model.

On first launch, Mirid downloads the runtime archive, verifies its exact size and SHA-256 hash, then installs it in the user's application-data directory. Later Mirid releases can update this runtime independently of the desktop installer.

The current stable package is runtime v6. The default Windows x64 package includes:

- the Mirid backend and voice service, including the official Parakeet.cpp CPU transcription runtime;
- current `llama.cpp` runners for CPU, Vulkan, and CUDA 12;
- CUDA-enabled `llama-cpp-python` and `stable-diffusion-cpp-python` builds;
- the Python and native dependencies required by Mirid's default feature set.

`runtime-release.json` is the machine-readable stable-channel manifest. Mirid pins each asset by filename, byte size, and SHA-256 digest before installation.

## Open models connected to Mirid

The linked model badges on the [Mirid Runtime Registry](https://huggingface.co/spaces/boneylizardwizard/mirid-runtime-registry) are generated from this audited set of live Hugging Face repositories.

- **Speech recognition:** [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo), [`UsefulSensors/moonshine-streaming-tiny`](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny), [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [`nvidia/stt_zh_conformer_transducer_large`](https://huggingface.co/nvidia/stt_zh_conformer_transducer_large), [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) and [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf).
- **Text to speech and voice conversion:** [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox), [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo), [`ResembleAI/chatterbox-nano`](https://huggingface.co/ResembleAI/chatterbox-nano), [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) and [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M).
- **Vision and structured image reading:** [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), [`LiquidAI/LFM2.5-VL-450M-Extract`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-Extract), [`LiquidAI/LFM2.5-VL-1.6B`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B), [`LiquidAI/LFM2.5-VL-1.6B-Extract`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) and [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).

## Hugging Face plumbing

Behind those model integrations, Mirid uses the Hub API for repository metadata and verified downloads. These are implementation details, not part of the linked model catalogue.

Model availability, licences and access conditions are governed by each model repository. A repository named here is an integration point, not a claim that its weights are included in every runtime download.

Source code is available in the [Mirid source repository](https://github.com/boneylizard/Eloquent) under the GNU Affero General Public License v3. The repository URL still carries the project's former name. Bundled third-party components remain subject to their respective licences.
