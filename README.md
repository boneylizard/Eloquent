# 🎭 Eloquent

**A local-first AI companion for roleplay, creative writing, and conversation.**

Eloquent is a feature-rich frontend for running large language models locally on your own hardware. It combines powerful LLM inference with text-to-speech, image generation, memory systems, and tools designed specifically for immersive roleplay and creative writing.

![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

---

## 🖼️ Screenshots

### Main Chat Interface

![Eloquent Chat Interface](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/chat%20new.png?raw=true)

*The main chat window showcasing all features: character selection, story tracker, choice generator, TTS controls, and more.*

---

## ✨ Features

### 🧠 Local LLM Inference
- **llama-cpp-python** backend for GGUF model support
- Multi-GPU tensor splitting for large models
- OpenAI-compatible API for external model providers
- Streaming responses with real-time token generation

### 🎤 Text-to-Speech

![TTS Settings](https://github.com/boneylizard/Eloquent/blob/main/eloquent%20launch/audio.jpg?raw=true)

*Comprehensive TTS controls with voice cloning, speed adjustment, and multiple engine support.*

- **Kokoro TTS** - Fast, high-quality neural TTS with multiple voices
- **Chatterbox TTS** - Voice cloning from audio samples
- Streaming TTS playback synchronized with LLM output
- Adjustable speed, pitch, and voice settings
- Real-time audio generation with low latency

### 🎭 Roleplay & Characters
- **Auto Character Creator** - Generate complete character profiles from conversation context
- Custom character creation with personality, backstory, and example dialogues
- **Story Tracker** - Track characters, inventory, locations, and plot points
- **Choice Generator** - Generate contextual action choices for interactive fiction
- **Author's Note** - Inject writing style guidance into any conversation
- Lore/world-building document support
- Automatic character portrait generation with Stable Diffusion

### 🎯 Focus Mode

![Focus Mode](https://raw.githubusercontent.com/boneylizard/Eloquent/refs/heads/main/eloquent%20launch/focus%20mode%20new.png)

*Distraction-free writing environment for immersive creative sessions.*

Focus Mode provides a clean, minimal interface that removes all distractions, letting you focus entirely on your conversation. Perfect for long-form creative writing and deep roleplay sessions.

### 🔍 Forensic Linguistics
- Analyze writing patterns and authorship
- Build text corpora from documents or web scraping
- Embedding-based similarity analysis (BGE-M3, Roberta)
- Compare writing samples for stylistic matches

### 🧩 Memory System
- Long-term memory extraction and retrieval
- User profile learning
- Context-aware memory injection

### 🎨 Image Generation
- Stable Diffusion integration (Automatic1111 API)
- In-chat image generation
- ADetailer support for face enhancement
- Character portrait generation

### 🔧 Additional Features
- Web search integration (DuckDuckGo)
- RAG (Retrieval-Augmented Generation) for document Q&A
- Code editor with syntax highlighting
- Anti-repetition system to reduce boilerplate responses
- Model testing and comparison tools

---

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (64-bit)
- **Python 3.12**
- **NVIDIA GPU** with CUDA support (8GB+ VRAM recommended)
- **Node.js 18+** for the frontend

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/boneylizard/Eloquent.git
   cd Eloquent
   ```

2. **Run the installer**
   
   Double-click **`install.bat`** or run it from the command line:
   ```bash
   install.bat
   ```
   
   This automatically:
   - Creates a Python virtual environment
   - Installs all Python dependencies (including pre-built wheels for tricky packages)
   - Installs frontend dependencies

3. **Download a GGUF model**
   
   Place your model files in a directory and configure the path in Settings when you first run Eloquent.

### Running Eloquent

Simply double-click **`run.bat`** to start everything!

Or from the command line:
```bash
run.bat
```

This will launch:
- Backend API server on `http://localhost:8000`
- TTS service on `http://localhost:8002`  
- Frontend at `http://localhost:5173` (opens automatically in your browser)

#### Debug Mode

For troubleshooting with visible console windows:
```bash
run_debug.bat
```

---

## ⚙️ Configuration

### Model Setup
1. Open Eloquent in your browser
2. Go to **Settings** → **Model Settings**
3. Set your model directory path
4. Select a model from the dropdown

### TTS Setup
1. Go to **Settings** → **Audio Settings**
2. Choose between **Kokoro** or **Chatterbox** TTS engine
3. For Chatterbox voice cloning, upload a voice reference audio file
4. Adjust speed, pitch, and other voice parameters

### External API (Optional)
Eloquent can connect to external OpenAI-compatible APIs:
1. Go to **Settings** → **API Endpoints**
2. Add your endpoint URL and API key
3. Select the endpoint in the model selector

---

## 🏗️ Project Structure

```
Eloquent/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI backend
│       ├── model_manager.py     # LLM loading and inference
│       ├── tts_service.py       # Text-to-speech
│       └── ...
├── frontend/
│   └── src/
│       ├── components/          # React components
│       ├── contexts/            # State management
│       └── utils/               # Helper functions
├── wheels/                      # Pre-built Python wheels
├── install.bat                  # One-click installer
├── run.bat                      # One-click launcher
├── run_debug.bat               # Debug mode launcher
└── requirements.txt
```

---

## 🎮 Usage Tips

### For Roleplay
- Create detailed character cards with personality traits and example dialogues
- Use the **Auto Character Creator** to generate profiles from existing conversations
- Use the **Story Tracker** to maintain consistency across long sessions
- Enable **Anti-Repetition Mode** in settings to reduce formulaic responses
- Use **Author's Note** to guide the AI's writing style for specific scenes
- Enable **Focus Mode** for distraction-free immersive sessions

### For Creative Writing
- Use the **Choice Generator** for branching narrative options
- Enable web search for research-assisted writing
- Upload reference documents for RAG-enhanced responses
- Use **Focus Mode** to eliminate distractions during writing sessions

### For Voice
- Kokoro is faster for general use
- Chatterbox provides voice cloning - upload a 3+ second audio sample
- Enable **Auto-Play TTS** for hands-free conversation
- Adjust speed and pitch to match character personalities

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

This project is licensed under the **GNU Affero General Public License v3.0** - see the [LICENSE](License) file for details.

---

## 🙏 Acknowledgments

Eloquent builds on many amazing open-source projects:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) & [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)

---

## 📬 Contact

Created by **Bernard Peter Fitzgerald** ([@boneylizard](https://github.com/boneylizard))

---

*Eloquent - Your local AI companion with limitless possibilities*
