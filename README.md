# Eloquent: Advanced Local AI Interface

Eloquent is a sophisticated, locally-hosted frontend for open-weight Large Language Models, designed around a dual-GPU architecture, advanced memory systems, and multi-modal AI interaction. It provides a powerful and private environment for interacting with, testing, and managing large language models.

## Key Features

### Dual-GPU Architecture
Eloquent is designed to leverage a dual-GPU setup for optimal performance. One GPU is dedicated to primary model inference, while the second GPU handles memory-related tasks, ensuring smooth and responsive interactions even with large models.

### Intelligent Memory System
The application features a persistent memory system that analyzes conversations and stores relevant information about user preferences, expertise, and context. This allows for truly personalized AI responses that improve over time.

### Advanced Chat & Character System

- **Customizable AI Personas**: Create and customize AI characters with detailed backgrounds, personalities, and speech patterns.
- **Dynamic Conversation Management**: Enjoy features like auto-generated chat titles, editable messages, and a full conversation history.
- **World Lore Integration**: Build rich, consistent fictional worlds with keyword-triggered contextual injection from a custom lore book.

### Comprehensive Document Intelligence (RAG)

- Upload and parse a wide range of document types (PDF, DOCX, TXT, etc.).
- Eloquent intelligently chunks and embeds document content for efficient retrieval.
- Relevant document chunks are automatically injected into the conversation context.

### Model ELO Testing System

- **Single Model Testing**: Evaluate the performance of individual models with automated LLM judges providing scores and feedback.
- **Comparison Testing**: Pit models against each other in head-to-head competitions with ELO rating calculations.
- **Advanced ELO Algorithm**: The system uses a category-weighted, performance-scaled rating system to provide a nuanced view of model capabilities.

### Multi-modal Interaction

- **Image Generation**: Generate images directly within the chat interface using either AUTOMATIC1111 or the built-in "EloDiffusion" engine powered by stable-diffusion.cpp.
- **ADetailer Integration**: Automatically enhance generated images with face and detail correction using ADetailer.
- **Voice-to-Text and Text-to-Speech**: Interact with the AI using your voice and receive spoken responses.

### Focus and Call Modes

- **Focus Mode**: A clean, distraction-free chat interface for immersive conversations.
- **Call Mode**: An audio-only interaction mode for a more natural, hands-free experience.

---

## Installation (Windows)

### Prerequisites

- A Windows operating system.
- Node.js version 21.7.3 is required for guaranteed compatibility.
- An NVIDIA GPU is highly recommended for full feature support.

### Setup

1. Clone or download the repository to your local machine.
2. Run the `install.bat` script located in the root directory. This will install all necessary Python and Node.js dependencies.
3. Once the installation is complete, run the `run.bat` script to launch the Eloquent application.

---

## 📞 Call Mode (Parakeet TTS/ASR) Setup

To enable Call Mode (real-time voice input and speech playback), Eloquent uses NVIDIA’s nemo-toolkit.  
Due to complex dependencies, it must be installed after the main setup:

```bash
pip install nemo-toolkit==2.3.1
```

---

## Cross-Platform Compatibility

While Eloquent was developed on Windows, the core technologies (Python, FastAPI, React) are cross-platform. However, running it on other operating systems requires manual setup.

### 🐧 Linux

- **Compatibility**: High  
- **Notes**:
  - Eloquent can run very effectively on Linux with an NVIDIA GPU, but requires manual setup.
  - The `install.bat` and `run.bat` scripts are for Windows only. You must create equivalent `.sh` shell scripts to perform the installation and launch steps.
  - You will need to manually install system-level dependencies like `espeak-ng` for the TTS service:
    ```bash
    sudo apt-get install espeak-ng
    ```
  - Full functionality assumes an NVIDIA GPU and the appropriate CUDA drivers.

### 🍎 macOS

- **Compatibility**: Low / Not Recommended  
- **Limitations**:
  - The application is designed for a dual-GPU setup using NVIDIA's CUDA platform. Most Macs, especially modern ones with Apple Silicon (M1/M2/M3), do not have NVIDIA GPUs.
  - The application would likely fall back to a very slow CPU-only mode, defeating its primary purpose.
  - macOS cannot run `.bat` files.
  - Manual installation of dependencies like `espeak-ng` (via Homebrew) would be required.

---

## Technical Architecture

Eloquent is built with a React frontend and a Python backend using FastAPI. Here's a breakdown of the key components:

### Frontend (`/src`)

- **AppContext.jsx**: The core of the frontend, this context provider manages the application's state, including models, conversations, settings, and API interactions.
- **Chat.jsx**: The main chat interface component, responsible for displaying messages, handling user input, and integrating various features like character selection, RAG, and multi-modal interactions.
- **ModelTester.jsx**: This component provides the interface for the ELO testing system, allowing users to set up and run single-model or comparison tests.
- **SimpleChatImageButton.jsx**: Manages the image generation dialog and ADetailer integration.
- **FocusModeOverlay.jsx & CallModeOverlay.jsx**: These components provide the specialized interfaces for the Focus and Call modes.

### Backend (`/app`)

- **main.py**: The main FastAPI application file, defining API endpoints and managing the application's lifecycle.
- **inference.py**: Handles the core text generation logic, providing a unified interface for interacting with different model backends.
- **memory_intelligence.py**: The heart of the memory system, responsible for semantic retrieval, creation, and curation of memories.
- **tts_service.py**: This service provides text-to-speech functionality, supporting multiple TTS engines.
