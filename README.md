# Eloquent: Advanced Local AI Interface

Eloquent is a sophisticated, locally-hosted frontend for open-weight Large Language Models, designed around a dual-GPU architecture, advanced memory systems, and multi-modal AI interaction. It provides a powerful and private environment for interacting with, testing, and managing large language models with performance rivaling commercial AI platforms.

![Chat UI](https://github.com/boneylizard/Eloquent/blob/main/assets/chat.jpg)

## Key Features

### High-Performance Dual-GPU Architecture

Eloquent is engineered to maximize performance through intelligent GPU resource allocation. The primary GPU handles model inference with optimized speeds (approximately 37 tokens/second on Gemma 3 27B with RTX 3090), while the secondary GPU manages memory operations, RAG processing, and auxiliary tasks. This architecture ensures responsive interactions even with large 27B+ parameter models.

### OpenAI-Compatible API Integration

- **Inbound API Support**: Full compatibility with OpenAI-compatible APIs including KoboldCPP streaming across networks  
- **Seamless Model Mixing**: Combine local GGUF models with remote API endpoints in the same interface  
- **Network Streaming**: Real-time streaming support for distributed inference setups  

### Intelligent Memory System

The application features a sophisticated persistent memory system that analyzes conversations using semantic embeddings and stores contextual information about user preferences, expertise areas, and conversation patterns. The memory system automatically injects relevant context into conversations, enabling truly personalized AI responses that evolve and improve over time through iterative alignment.

### Advanced Chat & Character System

- **Customizable AI Personas**: Create and customize AI characters with detailed backgrounds, personalities, speech patterns, and behavioral traits. Characters support rich metadata including world lore, dialogue styles, and memory preferences.  
![Characters UI](https://github.com/boneylizard/Eloquent/blob/main/assets/characters.jpg)

- **Dynamic Conversation Management**: Features include auto-generated chat titles using AI analysis, fully editable message history, conversation export/import, and intelligent context management  
- **World Lore Integration**: Build rich, consistent fictional worlds with a comprehensive lore book system  
- **Multi-Model Conversations**: Engage multiple models simultaneously in structured debates, comparisons, or collaborative discussions  

### Auto-Character Creator

Eloquent's revolutionary Auto-Character Creator streamlines character development through natural language interaction and intelligent JSON structure filling:

- Engage in natural dialogue with an AI about your character concept  
- Click Auto-Generate to trigger intelligent character sheet creation  
- The system analyzes conversation context and fills out comprehensive character data  
- **Smart Improvisation**: Fills gaps with consistent, creative details  
- **Iterative Refinement**: Modify characters using natural prompts  
- **Automated Avatar Generation**: Built-in Stable Diffusion prompt generation  
- **One-Click Deployment**: Saves characters directly to your permanent library  

![Auto-Character Creator](https://github.com/boneylizard/Eloquent/blob/main/assets/autocharacter%20creator.jpg)

### Comprehensive Document Intelligence (RAG)

- **Multi-Format Support**: PDF, DOCX, TXT and more  
- **Advanced Chunking**: Semantic segmentation for precision retrieval  
- **Embedding-Based Retrieval**: Using state-of-the-art embedding models  
- **Dynamic Context Injection**: Up to 5 chunks automatically injected  
- **Performance Optimization**: RAG runs on secondary GPU for speed  

### Advanced Model ELO Testing System

Eloquent provides research-grade model evaluation capabilities with sophisticated testing frameworks:

- **Single Model Testing**: LLM judges, feedback, scoring  
- **Head-to-Head Comparisons**: ELO calculations with statistical analysis  
- **Parameter Optimization**: Systematic testing across temperature, top-p, top-k, etc.  
- **Multi-Judge Validation**: Disagreement detection and consensus scoring  
- **Category-Weighted ELO**: Skill domain-specific ratings  
- **Auto-Suggested Analysis Questions**: Explore model behavior deeply  
- **Export/Import Results**: For collaboration and replication  
- **Analysis Chat**: AI-powered review of test outcomes  

![Model Tester](https://github.com/boneylizard/Eloquent/blob/main/assets/elo%20tester.jpg)

### Cutting-Edge Multi-modal Interaction

- **Local Image Generation**: SD, SDXL, Flux via stable-diffusion.cpp  
- **AUTOMATIC1111 Integration**: Seamless compatibility with A1111 workflows  
- **ADetailer Support**: Built-in face/detail enhancement  
- **Vision Models**: Upload images for model analysis  
- **Generate → Analyze Workflows**: Image-to-text pipelines  

### Push-to-Talk Voice Interaction

- **Reliable Control**: Shift key for start/stop recording  
- **Low-Latency Pipeline**: Fast ASR + TTS loop  
- **High Accuracy ASR**: Noise filtered, speaker adapted  
- **Text-to-Speech Output**: Quality voice playback  

### Focus and Call Modes

- **Focus Mode**: Minimalist interface for immersive, distraction-free conversation  
![Focus Mode](https://github.com/boneylizard/Eloquent/blob/main/assets/focus.jpg)

- **Call Mode**: Voice-first interface with real-time audio and avatar feedback  
![Call Mode](https://github.com/boneylizard/Eloquent/blob/main/assets/call.jpg)

### Cross-Platform Character Compatibility

- **Universal Format**: PNG files embed all character metadata  
- **SillyTavern Integration**: Full compatibility  
- **Kobold Support**: JSON export/import  
- **Metadata Preservation**: Portraits act as character containers  

---

## Installation (Windows)

### Prerequisites

- Windows 10/11  
- Node.js 21.7.3  
- NVIDIA GPU (dual-GPU ideal)  
- 16GB+ RAM  
- CUDA-compatible drivers  

### Setup Instructions

1. Clone or download this repo  
2. Run `install.bat`  
   - Installs Python dependencies  
   - Builds frontend  
   - Configures backend and initializes systems  
3. Run `run.bat` to launch  
4. Navigate to `http://localhost:5173` in your browser  

---

### 📞 Call Mode (Advanced Voice Features)

To enable Call Mode:

```bash
pip install nemo-toolkit==2.3.1
```

This enables high-quality push-to-talk interaction, TTS synthesis, and optimized ASR.

---

## Cross-Platform Compatibility

### 🐧 Linux

- Manual setup required  
- Equivalent `.sh` scripts needed  
- Install audio dependencies:
```bash
sudo apt-get install espeak-ng portaudio19-dev
```
- CUDA drivers mandatory for GPU acceleration  

### 🍎 macOS

- CPU-only fallback  
- Not recommended due to lack of CUDA support  
- Manual dependency installation via Homebrew  

---

## Technical Architecture

### Frontend Highlights

- `AppContext.jsx`: Central state management  
- `Chat.jsx`: Main chat interface  
- `ModelTester.jsx`: ELO testing interface  
- `AnalysisChat.jsx`: Post-test model review  
- `SimpleChatImageButton.jsx`: Image generation controls  
- `FocusModeOverlay.jsx`, `CallModeOverlay.jsx`: Custom UI overlays  

### Backend Highlights

- `main.py`: FastAPI server  
- `inference.py`: Streaming model inference  
- `memory_intelligence.py`: Long-term memory system  
- `tts_service.py`: TTS playback  
- `model_manager.py`: GPU-aware model loader  
- `rag_utils.py`: Document chunking and retrieval  

---

## Performance Specifications

- **Text Gen Speed**: ~37 tokens/sec (Gemma 3 27B on RTX 3090)  
- **ASR + TTS**: Near real-time with optimized latency  
- **Concurrent Tasks**: Dual-GPU architecture supports simultaneous operations  
- **Model Format Support**: Optimized for **GGUF** models  
- **API Support**: Full OpenAI-compatible inbound streaming  

---

## Contributing

We welcome contributions for:

- Model backend improvements  
- UI/UX enhancements  
- Multi-modal tools  
- Linux/macOS setup scripts  
- Advanced testing frameworks  

Please respect the dual-GPU design and performance standards that define Eloquent’s identity.

---
