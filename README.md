# Eloquent

**A highly-modular local-first AI platform with limitless versatility**

Eloquent is an extensible AI orchestration layer for your own hardware.  
It gives you a unified way to run and combine **local GGUF models**, **external API models**, **TTS/STT services**,  
**image generation**, **memory/RAG**, and **specialized analysis tools** behind a single interface and API.

Out of the box it ships with a full desktop console (React + FastAPI) and a set of "first-party modules":
LLM chat, model testing and judging, multi-GPU unified models, memory agents, document RAG, forensic linguistics,
voice pipelines, character tools, and more. But the architecture is deliberately open-ended –  
you can bolt on your own agents, routes, and UI panels without fighting the core system.

![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11–3.12-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

## 🖼️ Screenshots

### Main Chat Interface

![Eloquent Chat Interface](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/chat%20new.png?raw=true)

*The primary console with chat, story tracker, choice generator, author's note, TTS controls, and all control panels.*

### Audio / TTS Settings

![TTS Settings](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/audio.jpg?raw=true)

*Comprehensive audio control center with Kokoro, Chatterbox voice cloning, and streaming playback configuration.*

### Focus Mode

![Focus Mode](https://raw.githubusercontent.com/boneylizard/Eloquent/refs/heads/main/eloquent%20launch/focus%20mode%20new.png)

*Distraction-free interface for deep sessions, built on top of the same underlying model stack.*

### Character Library

![Character Library](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/characters.jpg?raw=true)

*Rich character library with profiles, portraits, and management tools built on top of the platform.*

### Model Elo Tester

![Model Elo Tester](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/elo%20tester.jpg?raw=true)

*Model versus model testing, ranking, and comparison UI running against the same backend orchestration.*

---

## ✨ Core Capabilities

### 🧠 LLM Orchestration & Multi-GPU

- **Local GGUF via `llama-cpp-python`**
  - Load models per-GPU or in unified multi-GPU mode
  - Tensor splitting with configurable `tensor_split` for 2, 3, 4+ GPUs
  - Dynamic GPU detection and allocation
- **API + Local Hybrid**
  - OpenAI-compatible API endpoints (Chub.ai, custom gateways, etc.) live alongside local models
  - Use API for primary chat while running smaller local models for memory, tools, or evaluation
  - Separate API model selector in the UI, independent of GPU-based local model selection
- **Model Manager & Services**
  - `ModelManager` and `ModelService` coordinate:
    - Per-GPU model allocation
    - Split-services vs unified-model modes
    - Special "purpose slots" (judge models, forensic embeddings, automation interpreters, etc.)
  - OpenAI-compatible router so other tools can talk to Eloquent like an OpenAI server
- **Streaming & Real-time**
  - Streaming responses with real-time token generation
  - WebSocket support for live updates

### 🔊 Voice & Audio Stack

- **Kokoro TTS**
  - Fast, high-quality neural TTS for general dialogue and narration
  - Multiple built-in voices
- **Chatterbox TTS**
  - Voice cloning from reference audio samples
  - Chunked streaming pipeline tuned to avoid cache overflows and latency spikes
  - Real-time audio generation with low latency
- **Dedicated TTS Service**
  - Separate FastAPI `tts_backend` service on its own port
  - `TTSClient` used from main backend and frontend for streaming playback
  - Auto-play TTS pipelines that synthesize and queue audio as LLM responses stream in
- **Optional STT**
  - `stt_service` for speech-to-text integration

### 🧩 Memory, RAG & Web Intelligence

- **Memory System**
  - Long-term memory extraction and retrieval
  - User profile learning and context injection
  - Memory routes for introspection and control
  - Memory intelligence agents that process conversation context
- **RAG (Retrieval-Augmented Generation)**
  - Document ingestion, embeddings, and retrieval via `rag_utils` and `Document_routes`
  - RAG status and controls exposed in the UI
  - Document store management and querying
- **Web Search Integration**
  - DuckDuckGo-style web search via `web_search_service`
  - Optional online grounding for answers

### 🔍 Forensic Linguistics & Analysis

- **ForensicLinguisticsService**
  - Pluggable pool of embedding models (BGE-M3, GTE, RoBERTa, Jina, Nomic, and many others as configured)
  - Build and cache corpora from documents or scraped text
  - Stylistic similarity and authorship-style comparison
  - Embedding-based similarity analysis
- **Analysis Panels**
  - Dedicated UI for running forensic jobs, watching progress, and inspecting results
  - Compare writing samples for stylistic matches

### 🎨 Image & Visuals

- **Built-in Stable Diffusion (stable_diffusion_cpp)**
  - Ships with pre-built `stable_diffusion_cpp_python` wheels
  - Runs fully local, no external API or web UI required
  - Used for in-app character portrait generation and other image workflows
  - Integrated directly into the backend via `sd_manager`
- **Optional Automatic1111 Integration**
  - Can connect to an existing Automatic1111 instance if you already use it
  - Treated as a secondary/optional path; the primary flow is the in-process Stable Diffusion backend
- **Image Management**
  - Generated images stored and exposed in the UI
  - Hooks available to extend or route images into your own pipelines

### 🎭 Character, Story & Interaction Modules

These are *verticals built on the platform*, not the platform itself:

- **Character Library**
  - Rich character cards with personality, background, and sample dialogue
  - Portraits generated via built-in Stable Diffusion
  - Library UI for browsing, editing, and switching personas
- **Auto Character Creator**
  - Generate complete character JSON from conversation transcripts
  - Works with both local models and external API models via unified backend route
- **Story Tracker**
  - Track characters, locations, items, flags, and plot state
  - Tracker data wired into the system prompt so the model actually uses it
- **Choice Generator**
  - Context-aware action/choice generation for interactive scenarios
  - Integrates with Story Tracker and chat context
- **Author's Note**
  - Inject writing style guidance into any conversation
  - Persists across sessions and injects into system prompt for both local and API models

### 🎯 Interaction UX & Tools

- **Main Chat Console**
  - Streaming responses, TTS auto-play pipelines, flexible context controls
  - Author's Note that injects instructions into system prompt for both local and API models
  - Anti-repetition system to reduce boilerplate responses
- **Focus Mode**
  - Minimal UI built on top of the same backend stack
  - Distraction-free environment for deep sessions
- **Model Tester & Elo-Style Comparison**
  - Load multiple models (local and/or API) and compare outputs side-by-side
  - Model Elo testing UI for structured evaluations
  - Judge models for automated quality assessment
- **Forensic & RAG Panels**
  - Dedicated surfaces for higher-level analysis and retrieval experiments
- **Code Editor Overlay**
  - In-app code view/editor layered over the main experience for power users

---

## 🏗️ Architecture Overview

At a high level:

- **Backend (`backend/app/`)**
  - `main.py` – FastAPI app and API router
  - `model_manager.py`, `model_service.py`, `model_subprocess.py` – LLM lifecycle, GPU orchestration, worker processes
  - `openai_compat.py` – OpenAI-style endpoints for external tools
  - `tts_backend.py`, `tts_service.py`, `tts_client.py` – TTS microservice and utilities
  - `stt_service.py` – Speech-to-text integration
  - `memory_intelligence.py`, `memory_agent.py`, `memory_routes.py` – Memory and RAG core
  - `Document_routes.py`, `rag_utils.py` – Document ingestion and retrieval
  - `forensic_linguistics_service.py` – Authorship and style analysis
  - `character_intelligence.py` – Character JSON generation (local + API)
  - `sd_manager.py` – Built-in Stable Diffusion integration
  - `web_search_service.py` – External search module
  - `inference.py` – Core LLM generation logic

- **Frontend (`frontend/src/`)**
  - `components/` – `Chat`, `StoryTracker`, `ChoiceGenerator`, `ForensicLinguistics`, `ImageGen`, `ModelSelector`, `ModelTester`, `Settings`, `Sidebar`, `Navbar`, `FocusModeOverlay`, `CharacterManager`, `CharacterEditor`, and many more
  - `contexts/` – `AppContext` orchestrates global state: active model, API vs local, TTS, memory, tracker, etc.
  - `utils/` – Utilities like anti-repetition processing and helpers shared across modules

- **Launch & Environment**
  - `install.bat` – One-click installer for Python env, wheels, backend, and frontend dependencies
  - `run.bat` – One-click launcher for backend(s) + frontend
  - `run_debug.bat` – Debug launcher with visible console windows
  - `launch.py` – Python launcher that handles GPU detection and service orchestration

---

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support  
  - VRAM requirements depend on your chosen models and GPU configuration
- **Python 3.11 or 3.12**
  - Must be compatible with the bundled `llama_cpp_python` and `stable_diffusion_cpp_python` wheels
  - The provided `install.bat` currently targets 3.11/3.12 specifically
- **Node.js v21.7.3** (strongly recommended)
  - Other versions may work but are not officially supported

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/boneylizard/Eloquent.git
   cd Eloquent
   ```

2. **Run the installer**

   Double-click **`install.bat`** in Explorer, or run it from a terminal:

   ```bash
   install.bat
   ```

   This will:

   - Create a Python virtual environment (`venv`)
   - Install PyTorch with CUDA 12.1
   - Install pre-built wheels for `llama_cpp_python` and `stable_diffusion_cpp_python`
   - Install backend Python dependencies from `requirements.txt`
   - Install frontend dependencies via `npm install`

3. **Download a GGUF model**

   - Place your `.gguf` models in a directory of your choice
   - You'll point Eloquent to this directory in **Settings → Model Settings** after the first run

### Running Eloquent

- **Normal mode**

  Double-click **`run.bat`**, or from a terminal:

  ```bash
  run.bat
  ```

  This will start:

  - Backend API server on `http://localhost:8000`
  - Dedicated TTS service on `http://localhost:8002`
  - Frontend dev server on `http://localhost:5173` (opens automatically in your browser)

- **Debug mode**

  For troubleshooting with visible console windows:

  ```bash
  run_debug.bat
  ```

---

## ⚙️ Configuration

### Model Setup

1. Open Eloquent in your browser
2. Go to **Settings → Model Settings**
3. Set your **model directory path** (where your GGUF files live)
4. Use the **Model Selector** to:
   - Choose local models for each GPU
   - Choose an **API model** separately (or run API-only)

The model selector UI reflects:

- **Per-GPU status** (which model is loaded where)
- Whether a slot is running a **local** model or an **API** endpoint
- Current GPU count and dynamic `tensor_split` configuration for unified models

### Multi-GPU & Tensor Split

- Eloquent supports:
  - **Split-services mode**: Different models on different GPUs for different roles
  - **Unified-model mode**: One large model sharded across 2, 3, 4+ GPUs
- In **Settings → Advanced / Tensor Split**:
  - View the current `tensor_split`
  - Provide a comma-separated list of values (e.g. `1,1,1,1` for four GPUs)
  - Values are normalized to sum to 1.0 and stored for future loads

### TTS Setup

1. Go to **Settings → Audio / TTS**
2. Choose:
   - **Kokoro** for fast, robust TTS
   - **Chatterbox** for voice cloning
3. For Chatterbox:
   - Upload a reference voice sample
   - Adjust speed and other parameters
4. Enable **Auto-Play TTS** if you want streaming playback as responses arrive

### External API Models

Eloquent can talk to any OpenAI-compatible API:

1. Go to **Settings → API Endpoints**
2. Add your endpoint URL, API key, and preferred model name
3. In the **Model Selector**, choose an API entry as the **API Model**
4. You can:
   - Use the API model for main chat
   - Still load local models for memory, tools, or judging

### Memory, RAG & Forensics

- **Memory & RAG**
  - Configure memory behavior and document ingestion in **Settings → Memory / RAG**
  - Use the RAG panels to upload documents and inspect retrieval behavior
- **Forensic Linguistics**
  - Load embedding models and run analyses in the **Forensic Linguistics** panel
  - Build corpora, compare samples, and run stylistic similarity jobs

---

## 🧪 Example Workflows

These are examples of what you *can* do with the platform using the built-in modules.

### Multi-Model Setup

- Load a large GGUF model in unified mode across multiple GPUs
- Load a second, smaller model on another GPU for:
  - Memory extraction
  - Judging / Elo testing
  - Tool-like tasks (summaries, classifiers)
- Attach an API model and compare it against your local stack using the Model Elo Tester

### Character-Driven Sessions

- Use the **Character Library** to pick or build a persona
- Enable **Story Tracker** so the system remembers characters, locations, and items
- Use **Choice Generator** to propose contextual actions
- Use **Author's Note** to steer style/behavior for this specific run without touching the global system prompt
- Generate character portraits using the built-in Stable Diffusion integration

### Analysis & Research

- Ingest documents into RAG and use **RAG Status / Settings** to tune behavior
- Use **Forensic Linguistics** to compare writing samples for stylistic similarity
- Toggle **web search** when you want external grounding

---

## 🤝 Contributing

Contributions, issues, and feature suggestions are welcome.  
If you build your own modules or verticals on top of Eloquent, feel free to share them.

This project is licensed under the **GNU Affero General Public License v3.0** – see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

Eloquent builds on many amazing open-source projects:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) & [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [stable-diffusion.cpp](https://github.com/adieyal/sd-cpp) & [stable-diffusion-cpp-python](https://github.com/adieyal/sd-cpp-python)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

---

## 📬 Contact

Created by **Bernard Peter Fitzgerald** ([@boneylizard](https://github.com/boneylizard))

---

*Eloquent – a highly-modular local-first AI platform with limitless versatility.*
