# Eloquent

**The most feature-complete local AI workstation. No subscriptions. No cloud dependency. Just your hardware.**

While everyone else ships another chat UI with fancy presets, Eloquent gives you **in-house Stable Diffusion**, **multi-GPU inference**, **voice cloning**, **model ELO testing**, **a tool-calling code editor**, **multi-role chat**, and **forensic linguistics** – all running locally.

Optional cloud APIs for when you want them. Your choice.

### What makes it different?
- **Single application**: LLM + image generation + voice + code tools + model evaluation
- **Multi-GPU that works**: Unified tensor splitting or dedicated GPU assignment
- **More than chat**: ELO testing framework, forensic linguistics, story state tracking, multi-role conversations
- **Production features**: Voice cloning, image upscaling, conversation summaries, agent mode

![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11–3.12-green.svg)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/pTcYDAUh)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

## ⚡ Quick Start

| I want to... | Do this |
|--------------|---------|
| **Chat with voice** | `install.bat` → `run.bat` → load a GGUF → enable Auto-TTS |
| **Generate images** | Drop `.safetensors` in a folder → Settings → Image Gen → set path |
| **Upscale images** | Generate image → click Upscale → select 2x/3x/4x |
| **Multi-character roleplay** | Settings → enable Multi-Role → add characters to roster |
| **Test models** | Model Tester → import prompts → run A/B with ELO ratings |
| **Edit code with AI** | Load Devstral → Code Editor → set project directory |
| **Clone a voice** | Settings → Audio → Chatterbox Turbo → upload reference |

---

## 👥 Who This Is For

**Power users with NVIDIA GPUs** who want a complete local AI stack instead of juggling 5 different tools.

**Roleplayers & writers** who need multi-character conversations, story state, portraits, and voice in one app.

**Model evaluators** who want ELO testing and judge orchestration without building research infrastructure.

**Privacy-first users** who don't want conversations leaving their machine.

### Not for you if:

- You don't have an NVIDIA GPU
- You're on Mac or Linux (Windows only)

---

## 🎯 Core Features

### Chat & Roleplay

**Multi-Role Conversations**
- Multiple characters in one chat with automatic speaker selection
- Per-character TTS voices and talkativeness weights
- Optional narrator with customizable interjection frequency
- User profile picker for switching between personas
- Group scene context for shared settings

**Story Management**
- **Story Tracker**: Characters, locations, inventory, objectives injected into AI context
- **Scene Summary**: Persistent context that grounds the AI in current mood and situation
- **Choice Generator**: Contextual actions with 6 behavior modes (Dramatic, Chaotic, Romantic, etc.)
- **Director Mode**: Toggle between character actions and narrative beats for plot steering
- **Conversation Summaries**: Save summaries and load them into fresh chats for continuity

**Standard Features**
- Character library and creator with AI-generated portraits
- Memory & RAG with document ingestion and web search
- Author's Note for direct AI guidance
- Focus Mode and Call Mode interfaces

### Inference & Models

**Multi-GPU Support**
- Unified tensor splitting across 2, 3, 4+ GPUs
- Split-services mode with dedicated GPU assignments
- Purpose slots for judge models and memory agents
- Real-time VRAM monitoring

**Model Compatibility**
- Local GGUF models via llama.cpp
- OpenAI-compatible APIs (OpenRouter, local proxies, Chub.ai)
- Simultaneous local + API model usage

### Image Generation

**Local Stable Diffusion**
- SD 1.5, SDXL, and FLUX support (safetensors/ckpt/gguf)
- Custom ADetailer with YOLO face detection and inpainting
- "Visualize Scene" - auto-generate images from chat context
- Set generated images as chat backgrounds

**Image Upscaling**
- Variable upscaling: 2x, 3x, 4x with ESRGAN models
- Model selector for different upscaler weights

**Cloud Fallback** *(Optional)*
- NanoGPT API for image generation without local GPU
- Experimental video generation (pay-per-use)

### Voice & Audio

**TTS Engines**
- **Kokoro**: Fast neural synthesis with multiple voices
- **Chatterbox**: Voice cloning from reference samples
- **Chatterbox Turbo**: Enhanced cloning with paralinguistic cues (`[laugh]`, `[sigh]`, `[cough]`)

**Features**
- Chunked streaming pipeline for low latency
- Auto-TTS with one-click toggle
- Call Mode: Full-screen voice conversation with animated avatars
- Per-character voice assignment in multi-role chat

### Model Evaluation

**ELO Testing Framework**
- Single model testing against prompt collections (MT-Bench, custom)
- A/B head-to-head comparisons with ELO updates
- Dual-judge mode with reconciliation
- Character-aware judging with custom evaluation criteria
- Parameter sweeps (temperature, top_p, top_k)
- 14 built-in analysis perspectives including 6-Year-Old Transformer Boy, Al Swearengen, Bill Burr, Alex Jones
- Import/export results with full metadata

### Code Editor

**Tool-Calling Agent**
- Devstral Small 2 24B (local) or Devstral Large (OpenRouter)
- File operations with automatic `.bak` backups
- Shell execution (optional, sandboxed)
- Vision support via screenshots

**Agent Mode Features**
- Chain of Thought visualization - see reasoning before actions
- Hallucination Rescue - executes intended tools even when JSON parsing fails
- Loop detection prevents endless file reading
- File explorer with full drive navigation

**Security**
- Sandboxed to working directory
- Optional command execution
- Automatic backups on file writes

### Analysis & Tools

**Forensic Linguistics**
- Authorship analysis and stylistic comparison
- Pluggable embedding models (BGE-M3, GTE, RoBERTa, Jina, Nomic)
- Build corpora from documents or scraped text

**UI & Customization**
- 5 premium themes: Claude, Messenger, WhatsApp, Cyberpunk, ChatGPT Light
- Text formatting: Quote highlighting, H1-H3 headings, paragraph controls
- Mobile-optimized with responsive layouts
- Auto-save settings (directories require manual save)

---

## 🖼️ Screenshots

### Main Chat
![Chat Interface](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/chat%20new.png?raw=true)
*Full chat with Story Tracker, Choice Generator, streaming TTS, and model control.*

### Audio Control
![TTS Settings](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/audio.jpg?raw=true)
*Voice cloning with real-time streaming playback.*

### Focus Mode
![Focus Mode](https://raw.githubusercontent.com/boneylizard/Eloquent/refs/heads/main/eloquent%20launch/focus%20mode%20new.png)
*Distraction-free interface.*

### Character Library
![Characters](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/characters.jpg?raw=true)
*AI-generated character portraits via built-in Stable Diffusion.*

### ELO Tester
![Elo Tester](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/elo%20tester.jpg?raw=true)
*Professional model evaluation with dual-judge reconciliation.*

### Mobile Themes
![Messenger](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/messengerchat.png?raw=true)
![WhatsApp](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/Whatsapplinemenu.png?raw=true)
![Cyberpunk](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/cyberpunkllmsettings.png?raw=true)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  Chat │ StoryTracker │ ChoiceGenerator │ ModelTester │ Settings │
│  ForensicLinguistics │ CodeEditor │ ImageGen │ MemoryEditor     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
│                                                                 │
│  ModelManager │ SDManager+ADetailer │ ForensicLinguistics      │
│  TTS Service │ Memory Intelligence │ Devstral Service          │
│  RAG Utils │ Web Search │ OpenAI Compat Layer                  │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────────┐
         │ GPU 0  │    │ GPU 1  │    │ Optional   │
         │ GGUF   │    │ GGUF   │    │ Cloud APIs │
         └────────┘    └────────┘    └────────────┘
```

**Stats**: 99 source files, 50,000+ lines of code  
**Backend**: 27 Python files (~21,000 lines)  
**Frontend**: 72 JSX files (~30,000 lines)

---

## 🚀 Installation

### Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support
- Python 3.11 or 3.12
- Node.js v21.7.3 (recommended)

### VRAM Guide

| Use Case | Recommended VRAM |
|----------|------------------|
| Small models (7B Q4) | 8GB |
| Medium models (13B-20B) | 12GB |
| Large models (70B+) | 24GB+ or multi-GPU |
| SD 1.5 | 4GB+ |
| SDXL/FLUX | 8GB+ |
| LLM + image gen together | 16GB+ or split across GPUs |

### Install & Run

```bash
git clone https://github.com/boneylizard/Eloquent
cd Eloquent
install.bat    # Wait for completion (5-10 minutes)
run.bat
```

The installer handles everything: Python venv, PyTorch with CUDA 12.1, pre-built wheels, all dependencies.

**Default ports:**
- Backend: `http://localhost:8000`
- TTS: `http://localhost:8002`
- Frontend: `http://localhost:5173`

Port conflicts are handled automatically - the frontend discovers actual ports.

---

## ⚙️ Configuration

### Models
1. Settings → Model Settings → set GGUF directory
2. Model Selector → choose per-GPU or unified multi-GPU
3. Add OpenAI-compatible API endpoints if desired

### Images
1. Settings → Image Generation → set safetensors directory
2. ADetailer Models → point to YOLO `.pt` files
3. Upscaler Models → point to ESRGAN `.pth` files

### Voice
1. Settings → Audio → choose Kokoro or Chatterbox/Chatterbox Turbo
2. For cloning: upload reference sample
3. Enable Auto-TTS toggle in chat

### Multi-Role
1. Settings → enable Multi-Role Chat
2. Click roster button → add characters
3. Set talkativeness weights and voices
4. Optionally enable narrator

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing dependencies | Run `install.bat` again or `pip install` the missing package |
| CUDA errors | Update NVIDIA drivers, ensure CUDA 12.x |
| Model won't load | Check VRAM, try smaller quantization |
| Port conflicts | Let auto-detection handle it or check processes on 8000/8002/5173 |
| TTS not working | Verify TTS service started in console |
| Import error on startup | Read traceback, activate venv, `pip install` the package |
| Context too large | Settings → adjust Context Window per endpoint |
| Agent Mode JSON errors | Hallucination Rescue should auto-fix, check console |

---

## 🧪 Example Workflows

**Multi-Character Roleplay**
- Enable Multi-Role in settings
- Add 3-4 characters to roster with different voices
- Set talkativeness weights (quiet character = 0.3, loud = 1.5)
- Enable narrator for scene-setting every 5 turns
- Use Story Tracker to maintain world state
- Generate portraits with built-in SD

**Model Evaluation**
- Import MT-Bench prompts
- Run A/B tests between two 70B models
- Enable dual-judge with Al Swearengen and Bill Burr
- Run parameter sweep on temperature
- Export results with ELO rankings

**Local Cursor Alternative**
- Load Devstral Small 2 24B
- Open Code Editor → set project directory
- Enable Chain of Thought to see reasoning
- Ask it to refactor a module
- Watch tool execution in real-time

**Long-Form Writing**
- Load character and enable Story Tracker
- Use Director Mode to steer plot beats
- Generate scene visualizations
- Save conversation summary every chapter
- Load summaries into fresh chats for continuity

---

## 📊 By The Numbers

| Metric | Value |
|--------|-------|
| Total lines of code | 50,000+ |
| Python backend files | 27 |
| React frontend files | 72 |
| Built-in TTS engines | 3 (Kokoro + Chatterbox + Chatterbox Turbo) |
| Analysis perspectives | 14 |
| Code editor tools | 7 |
| Supported SD architectures | 3 (SD 1.5, SDXL, FLUX) |
| Premium themes | 5 |

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and updates.

---

## 🤝 Contributing

Built by one developer with a day job. Contributions welcome.

Licensed under **GNU Affero General Public License v3.0**.

---

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) & [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) & [stable-diffusion-cpp-python](https://huggingface.co/boneylizard/stable_diffusion_cpp_python)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

---

## 📬 Contact

**Bernard Peter Fitzgerald** ([@boneylizard](https://github.com/boneylizard))

---

*Eloquent – your GPUs deserve better.*
