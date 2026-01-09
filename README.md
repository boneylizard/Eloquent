# Eloquent

**The most feature-complete local AI platform you've never heard of.**

While everyone else is fighting over which chat UI has the best preset system, Eloquent shipped with **in-house Stable Diffusion**, **multi-GPU inference**, **voice cloning**, **a full model ELO testing framework**, **a tool-calling code editor**, and **forensic linguistics analysis** – all running locally on your hardware.

Runs fully local. No subscriptions. No cloud dependency. APIs optional. Just your GPUs doing work.

### What makes it different?
- **Single app**: LLM + Stable Diffusion + voice cloning + model evaluation + code tools
- **Multi-GPU orchestration**: Unified tensor splitting or purpose-specific GPU slots
- **More than chat**: Built-in evaluation framework, forensic linguistics, story state tracking

![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11–3.12-green.svg)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/pTcYDAUh)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

## ⚡ Start Here

| I want to... | Do this |
|--------------|---------|
| **Chat + voice** | `install.bat` → `run.bat` → load a GGUF → enable TTS |
| **Generate images** | Drop `.safetensors` in a folder → Settings → Image Gen → set path |
| **Test models** | Open Model Tester → import prompts or MT-Bench → run A/B |
| **Use the code editor** | Load Devstral Small 2 24B (GGUF or API) → open Code Editor → point to a directory |
| **Clone a voice** | Settings → Audio → Chatterbox → upload reference sample |

---

## 👥 Who Is This For?

- **Power users with GPUs** who want an all-in-one local stack instead of 5 different tools
- **Roleplayers & writers** who want story state, choices, portraits, and voice in one place
- **Model evaluators** who want ELO testing and judge orchestration without setting up a research pipeline
- **Privacy-first users** who don't want their conversations leaving their machine

### Who it's *not* for (yet)

- Laptop users without dedicated GPU
- AMD GPU users (CUDA required)
- Mac users
- People who want a one-click mobile app experience

---

## 🎯 What's Inside

### Chat & Roleplay
- ✅ **Full-featured AI chat** – character library, character creator, personas, conversation management
- ✅ **Story Tracker** – characters, locations, inventory injected directly into AI context
- ✅ **Choice Generator** – contextual action options with full prompt expansion
- ✅ **Memory & RAG** – long-term memory, document ingestion, web search

### Inference
- ✅ **Multi-GPU inference** – unified tensor splitting or split-services mode across 2, 3, 4+ GPUs
- ✅ **OpenAI API compatible** – works with Chub.ai, local proxies, any compatible endpoint

### Image Generation
- ✅ **In-house Stable Diffusion** – drop in `.safetensors` files, supports SD 1.5, SDXL, and FLUX
- ✅ **Custom ADetailer** – YOLO face detection → mask generation → inpainting enhancement

### Voice
- ✅ **Voice cloning TTS** – Kokoro neural TTS + Chatterbox voice cloning with streaming playback
- ✅ **Call Mode** – full-screen voice conversation with speaking animations

### Evaluation & Analysis
- ✅ **Model ELO testing** – A/B comparisons, dual-judge reconciliation, parameter sweeps, 14 analysis perspectives
- ✅ **Forensic linguistics** – authorship analysis with pluggable embedding models

### Tools
- ✅ **Tool-calling code editor** – Devstral Small 2 24B with file ops and shell execution

**50,000+ lines of code. 99 source files. One hobby project.**

---

## 🖼️ Screenshots

### Main Chat Interface
![Eloquent Chat Interface](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/chat%20new.png?raw=true)
*Full-featured chat with Story Tracker, Choice Generator, Author's Note, streaming TTS, and complete model control.*

### Voice & Audio Control Center
![TTS Settings](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/audio.jpg?raw=true)
*Kokoro neural TTS + Chatterbox voice cloning with real-time streaming playback.*

### Focus Mode
![Focus Mode](https://raw.githubusercontent.com/boneylizard/Eloquent/refs/heads/main/eloquent%20launch/focus%20mode%20new.png)
*Distraction-free interface for deep work sessions.*

### Character Library
![Character Library](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/characters.jpg?raw=true)
*Rich character profiles with AI-generated portraits via built-in Stable Diffusion.*

### Model ELO Tester
![Model Elo Tester](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/elo%20tester.jpg?raw=true)
*Professional-grade model evaluation with dual-judge reconciliation, parameter sweeps, and character-based analysis perspectives.*

---

## 🔥 Flagship Features

### 🎨 In-House Stable Diffusion (No API, No External UI)

**A fully integrated local LLM + image generation stack. No external UI required.**

- Drop in any `.safetensors` file – SD 1.5, SDXL, or FLUX
- **Custom ADetailer implementation** – face detection → mask generation → inpainting enhancement
- YOLO-based face detection with configurable confidence thresholds
- Tuned mask parameters to prevent halos and harsh seams
- Generate character portraits directly in chat
- No Automatic1111, no ComfyUI, no external processes – it's all built-in

```
Your GPU → stable-diffusion.cpp → SDManager → ADetailerProcessor → Enhanced Image
```

### 🧠 Multi-GPU That Actually Works

Stop fighting with tensor parallelism configs that crash. Eloquent handles it.

- **Unified Mode**: Shard one large model across 2, 3, 4+ GPUs with automatic tensor splitting
- **Split Services Mode**: Different models on different GPUs for different tasks
- **Purpose Slots**: Dedicated GPU allocation for test models, judge models, memory agents
- Dynamic GPU detection and real-time VRAM monitoring
- Works with both local GGUF models AND OpenAI-compatible APIs simultaneously

### 🎯 Model ELO Testing Framework

**A complete model evaluation system that rivals research lab tooling.**

- **Single Model Testing**: Score responses against prompt collections (MT-Bench, custom)
- **A/B Comparison**: Head-to-head model battles with ELO rating updates
- **Dual-Judge Mode**: Two different models evaluate, then reconcile disagreements
- **Character-Aware Judging**: Judges can roleplay personas with custom evaluation criteria
- **Parameter Sweeps**: Test the same model across temperature/top_p/top_k combinations
- **14 Built-in Analysis Perspectives** including:
  - 🤖 6-Year-Old Transformer Boy (asks naive questions that expose assumptions)
  - 🥃 Al Swearengen from Deadwood (cuts through bullshit)
  - 😤 Bill Burr (angry skepticism)
  - 📢 Alex Jones (adversarial red-teaming)
  - 👩‍🏫 Dolores Umbridge (bureaucratic authority)
- Import/export full test results with all metadata
- Persistent ELO ratings across sessions

### 🎭 Interactive Roleplay Tools

**Story Tracker and Choice Generator that actually integrate with AI context.**

#### Story Tracker
- Track characters, locations, inventory, plot points
- Pin important items, mark things as starred
- Set current objectives
- **Data automatically injected into system prompts** – the AI actually knows your story state

#### Choice Generator
- Generate contextual action choices based on conversation + character + story state
- 6 behavior modes: Balanced, Dramatic, Subtle, Chaotic, Romantic, Action
- Custom direction input ("Include a choice where I confess my feelings")
- **Full prompt expansion**: Clicking a choice generates a complete first-person action, not just two words
- Edit any choice, regenerate individual options, continue from partial text

### 🔊 Voice Pipeline

**Real-time TTS that streams as the AI generates.**

- **Kokoro TTS**: Fast neural synthesis with multiple voices
- **Chatterbox TTS**: Voice cloning from reference samples
- Chunked streaming pipeline tuned for low latency
- Auto-play mode: Audio synthesizes and queues as tokens arrive
- **Call Mode**: Full-screen voice conversation interface with speaking animations
- Speech detection and push-to-talk support

### 💻 Tool-Calling Code Editor

**A local, tool-calling code editor in the Cursor/Windsurf spirit.**

Built for Devstral Small 2 24B with full tool-calling support:

| Tool | What It Does |
|------|--------------|
| `read_file` | Read file contents |
| `write_file` | Create/modify files (auto-backup) |
| `list_directory` | Browse folder structure |
| `search_files` | Grep-like content search |
| `run_command` | Execute shell commands |
| `create_directory` | Make folders |
| `delete_file` | Remove files |

- Path sandboxing (can't escape working directory)
- Vision support (screenshot → understanding)
- Session persistence
- Drag-and-drop files into context

> **Security notes**: Tool calls are restricted to the selected working directory. `run_command` is optional and can be disabled. `write_file` creates `.bak` backups automatically. No network access unless you explicitly enable web search.

### 🔬 Forensic Linguistics

**Authorship analysis and stylistic comparison.**

- Pluggable embedding models (BGE-M3, GTE, RoBERTa, Jina, Nomic, and more)
- Build corpora from documents or scraped text
- Compare writing samples for stylistic similarity
- Detailed analysis UI with progress tracking

### 🧩 Memory & RAG

- Long-term memory extraction and retrieval
- User profile learning
- Document ingestion with embeddings
- Web search integration (DuckDuckGo)
- Memory context injection into prompts

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
│                    Backend (FastAPI) - 6000+ lines              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ModelManager │  │ SDManager   │  │ ForensicLinguistics     │ │
│  │ Multi-GPU   │  │ +ADetailer  │  │ Service                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ TTS Service │  │ Memory      │  │ Devstral Service        │ │
│  │ Kokoro/CB   │  │ Intelligence│  │ Code Editor Tools       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ RAG Utils   │  │ Web Search  │  │ OpenAI Compat Layer     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────────┐
         │ GPU 0  │    │ GPU 1  │    │ External   │
         │ GGUF   │    │ GGUF   │    │ API        │
         └────────┘    └────────┘    └────────────┘
```

**Backend**: 27 Python files, ~21,000 lines  
**Frontend**: 72 JSX files, ~30,000 lines  
**Total**: 99+ files, 50,000+ lines of code

---

## 🚀 Quick Start

### Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support
- **Python 3.11 or 3.12**
- **Node.js v21.7.3** (recommended)

### VRAM Expectations

| Use Case | Recommended VRAM |
|----------|------------------|
| Small models (7B Q4) | 8GB |
| Medium models (13B-20B) | 12GB |
| Large models (70B+) | 24GB+ or multi-GPU |
| Image generation (SD 1.5) | 4GB+ |
| Image generation (SDXL/FLUX) | 8GB+ |
| Running LLM + image gen together | 16GB+ or split across GPUs |

### Installation

```bash
# Clone the repository
git clone https://github.com/boneylizard/Eloquent

# Move into the project directory
cd Eloquent

# Run the automated installer (Wait for this to finish!)
install.bat

# Launch the platform
run.bat
```

The installer handles everything:
- Python virtual environment
- PyTorch with CUDA 12.1
- Pre-built wheels for `llama_cpp_python` and `stable_diffusion_cpp_python`
- All Python dependencies
- Frontend npm packages

### Running

```bash
run.bat
```

Opens:
- Backend API: `http://localhost:8000`
- TTS Service: `http://localhost:8002`
- Frontend: `http://localhost:5173`

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Missing dependencies after install | Run `install.bat` again, or activate venv and `pip install` the package named in the traceback |
| CUDA errors on startup | Update NVIDIA drivers, ensure CUDA 12.x installed |
| Port already in use | Check for other processes on 8000/8002/5173 |
| Model fails to load | Check VRAM, try a smaller quantization |
| Windows Defender blocks | Add project folder to exclusions |
| TTS not working | Ensure `tts_backend` service started (check console) |

> **Note**: If the backend crashes on startup with an import error, read the traceback – it usually tells you exactly which package is missing. Activate the venv (`venv\Scripts\activate`) and `pip install` it.

---

## ⚙️ Configuration

### Models
1. **Settings → Model Settings** → Set your GGUF directory
2. **Model Selector** → Choose models per-GPU or enable unified multi-GPU mode
3. **API Endpoints** → Add OpenAI-compatible endpoints (Chub.ai, local proxies, etc.)

### Image Generation
1. **Settings → Image Generation** → Set your `.safetensors` directory
2. **ADetailer Models** → Point to your YOLO `.pt` files for face enhancement
3. Works with SD 1.5, SDXL, and FLUX models

### Voice
1. **Settings → Audio/TTS** → Choose Kokoro or Chatterbox
2. For voice cloning: Upload a reference sample
3. Enable Auto-Play for streaming synthesis

---

## 🧪 Example Workflows

### Multi-GPU Inference
- Load a 70B model across 4 GPUs in unified mode
- Run a smaller 7B model on a dedicated GPU for memory/tools
- Compare both against an API model using ELO tester

### Interactive Fiction
- Select a character from the library
- Enable Story Tracker to maintain world state
- Use Choice Generator for contextual action options
- Generate character portraits with built-in Stable Diffusion
- Enable voice mode for spoken dialogue

### Model Evaluation
- Import MT-Bench or custom prompt collections
- Run A/B tests between models
- Enable dual-judge mode with character perspectives
- Export results for analysis
- Track ELO ratings over time

### Local Cursor Alternative
- Load Devstral Small 2 24B
- Open the Code Editor overlay
- Point it at your project directory
- Ask it to read, modify, and create files
- It executes tool calls directly on your filesystem

---

## 📊 By The Numbers

| Metric | Value |
|--------|-------|
| Total lines of code | 50,000+ |
| Python backend files | 27 |
| React frontend files | 72 |
| Largest file (main.py) | 5,965 lines |
| Built-in TTS engines | 2 (Kokoro + Chatterbox) |
| Analysis perspectives | 14 |
| Tool definitions (code editor) | 7 |
| Supported SD architectures | 3 (SD 1.5, SDXL, FLUX) |

---

## 🤝 Contributing

This started as a hobby project by one developer working a day job. Contributions welcome.

Licensed under **GNU Affero General Public License v3.0**.

---

## 🙏 Acknowledgments

Built on the shoulders of giants:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) & [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) & [stable-diffusion-cpp-python](https://github.com/william-murray1204/stable-diffusion-cpp-python)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [ultralytics YOLO](https://github.com/ultralytics/ultralytics) (for ADetailer)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

---

## 📬 Contact

Created by **Bernard Peter Fitzgerald** ([@boneylizard](https://github.com/boneylizard))

---

*Eloquent – because your GPUs deserve better than another chat UI.*

