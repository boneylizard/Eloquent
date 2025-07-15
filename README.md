Local LLM Frontend
A customizable frontend for running large language models locally in GGUF format with Stable Diffusion integration, document processing, and RAG capabilities.
Features

Local LLM Execution: Run GGUF format LLMs directly on your hardware
Model Management: Easily load/unload different models
Image Generation: Integration with Automatic1111 WebUI or ComfyUI
Document Processing: Upload and process PDF, TXT, DOC, DOCX, and CSV files
Retrieval-Augmented Generation (RAG): Enhance responses with information from your documents
Memory-Efficient: Optimized for running on consumer hardware
Fully Offline: All processing happens locally, no data leaves your machine

System Requirements

CPU: Modern multi-core CPU
RAM: Minimum 16GB, 32GB+ recommended
GPU: NVIDIA GPU with at least 8GB VRAM (40GB VRAM optimal for larger models)
Storage: SSD with at least 10GB free space (more for storing multiple models)
OS: Windows, macOS, or Linux

Installation
Prerequisites

Python 3.9+ for the backend
Node.js 18+ for the frontend
[Optional] Automatic1111 WebUI for image generation

Backend Setup

Clone the repository:
bashCopygit clone https://github.com/yourusername/local-llm-frontend.git
cd local-llm-frontend

Create and activate a virtual environment:
bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashCopycd backend
pip install -r requirements.txt

Enable RAG in chat settings to use document knowledge in responses

Architecture
The application consists of two main components:
Backend (Python/FastAPI)

Model Management: Handles loading/unloading of GGUF models using llama-cpp-python or CTransformers
Text Generation: Processes prompts and generates text with the loaded model
Document Processing: Extracts text from documents, splits into chunks, and generates embeddings
Vector Database: Stores document embeddings for semantic search using FAISS
Image Generation: Communicates with Automatic1111 WebUI API

Frontend (React/Vite)

Chat Interface: Conversation UI with streaming responses
Model Manager: UI for loading/unloading models
Document Manager: Upload and manage documents for RAG
Image Generation: Interface for generating images with Stable Diffusion
Settings: Customize generation parameters and application preferences

Project Structure
Copylocal-llm-frontend/
├── backend/
│   ├── app.py                  # Main FastAPI application
│   ├── llm/                    # LLM handling
│   │   ├── model_manager.py    # Model loading/unloading
│   │   └── inference.py        # Text generation
│   ├── rag/                    # RAG functionality
│   │   ├── document.py         # Document processing
│   │   └── retrieval.py        # Context retrieval
│   ├── sd/                     # Stable Diffusion integration
│   │   └── auto1111.py         # Automatic1111 API client
│   └── utils/                  # Utility functions
├── frontend/
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── contexts/           # React contexts
│   │   └── App.jsx             # Main application
│   └── package.json            # Frontend dependencies
└── data/
    ├── models/                 # GGUF model storage
    ├── uploads/                # Original document uploads
    ├── processed/              # Processed documents and embeddings
    └── images/                 # Generated images
Configuration
Backend Configuration
Environment variables that can be set:

UPLOAD_DIR: Directory for document uploads (default: data/uploads)
DATA_DIR: Base directory for all data (default: data)
EMBEDDING_MODEL: Model to use for embeddings (default: all-MiniLM-L6-v2)
CHUNK_SIZE: Size of document chunks (default: 350)
CHUNK_OVERLAP: Overlap between chunks (default: 50)
AUTOMATIC1111_URL: URL for Automatic1111 WebUI (default: http://127.0.0.1:7860)

Frontend Configuration
Environment variables that can be set in .env file:

VITE_API_URL: URL of the backend API (default: http://localhost:8000)

Development
Adding New Features

Backend: Add new endpoints to app.py and implement functionality in appropriate modules
Frontend: Add new components in src/components and update the application state in contexts

Customizing UI

Modify Tailwind theme in tailwind.config.js
Update component styles in src/components/ui

Troubleshooting
Common Issues

Model Loading Fails: Ensure you have enough available VRAM/RAM for the model size
Image Generation Not Working: Check if Automatic1111 is running with the --api flag
Document Processing Slow: Large documents may take time to process, especially on CPU-only systems

Logs

Backend logs are printed to the console
Check browser console for frontend errors

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

llama-cpp-python for local LLM inference
AUTOMATIC1111/stable-diffusion-webui for image generation
sentence-transformers for document embeddings
FAISS for efficient similarity search
React and Tailwind CSS for the frontend Create necessary directories:
bashCopymkdir -p data/models data/uploads data/processed data/images


Frontend Setup

Install frontend dependencies:
bashCopycd frontend
npm install


Optional: Automatic1111 WebUI Setup

Follow the installation instructions at AUTOMATIC1111/stable-diffusion-webui
Launch with the --api flag to enable API access:
bashCopypython launch.py --api


Running the Application

Start the backend server:
bashCopycd backend
python app.py

Start the frontend development server:
bashCopycd frontend
npm run dev

Open your browser and navigate to http://localhost:3000

Model Setup

Download GGUF format models from Hugging Face
Place models in the data/models directory
Use the Models tab in the UI to load models

Usage
Chat

Load a model from the Models tab
Navigate to the Chat tab
Type your message and press Enter or click Send
Adjust generation settings as needed

Image Generation

Ensure Automatic1111 WebUI is running
Navigate to the Image tab
Enter a prompt and adjust settings
Click Generate

Document Processing and RAG

Navigate to the Documents tab
Upload documents (PDF, TXT, DOC, DOCX, CSV)
Select documents for RAG in the chat interface
Enable RAG in chat settings to use document knowledge in responses

Architecture
The application consists of two main components:
Backend (Python/FastAPI)

Model Management: Handles loading/unloading of GGUF models using llama-cpp-python or CTransformers
Text Generation: Processes prompts and generates text with the loaded model
Document Processing: Extracts text from documents, splits into chunks, and generates embeddings
Vector Database: Stores document embeddings for semantic search using FAISS
Image Generation: Communicates with Automatic1111 WebUI API

Frontend (React/Vite)

Chat Interface: Conversation UI with streaming responses
Model Manager: UI for loading/unloading models
Document Manager: Upload and manage documents for RAG
Image Generation: Interface for generating images with Stable Diffusion
Settings: Customize generation parameters and application preferences

Project Structure
Copylocal-llm-frontend/
├── backend/
│   ├── app.py                  # Main FastAPI application
│   ├── llm/                    # LLM handling
│   │   ├── model_manager.py    # Model loading/unloading
│   │   └── inference.py        # Text generation
│   ├── rag/                    # RAG functionality
│   │   ├── document.py         # Document processing
│   │   └── retrieval.py        # Context retrieval
│   ├── sd/                     # Stable Diffusion integration
│   │   └── auto1111.py         # Automatic1111 API client
│   └── utils/                  # Utility functions
├── frontend/
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── contexts/           # React contexts
│   │   └── App.jsx             # Main application
│   └── package.json            # Frontend dependencies
└── data/
    ├── models/                 # GGUF model storage
    ├── uploads/                # Original document uploads
    ├── processed/              # Processed documents and embeddings
    └── images/                 # Generated images
Configuration
Backend Configuration
Environment variables that can be set:

UPLOAD_DIR: Directory for document uploads (default: data/uploads)
DATA_DIR: Base directory for all data (default: data)
EMBEDDING_MODEL: Model to use for embeddings (default: all-MiniLM-L6-v2)
CHUNK_SIZE: Size of document chunks (default: 350)
CHUNK_OVERLAP: Overlap between chunks (default: 50)
AUTOMATIC1111_URL: URL for Automatic1111 WebUI (default: http://127.0.0.1:7860)

Frontend Configuration
Environment variables that can be set in .env file:

VITE_API_URL: URL of the backend API (default: http://localhost:8000)

Development
Adding New Features

Backend: Add new endpoints to app.py and implement functionality in appropriate modules
Frontend: Add new components in src/components and update the application state in contexts

Customizing UI

Modify Tailwind theme in tailwind.config.js
Update component styles in src/components/ui

Troubleshooting
Common Issues

Model Loading Fails: Ensure you have enough available VRAM/RAM for the model size
Image Generation Not Working: Check if Automatic1111 is running with the --api flag
Document Processing Slow: Large documents may take time to process, especially on CPU-only systems

Logs

Backend logs are printed to the console
Check browser console for frontend errors

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

llama-cpp-python for local LLM inference
AUTOMATIC1111/stable-diffusion-webui for image generation
sentence-transformers for document embeddings
FAISS for efficient similarity search
React and Tailwind CSS for the frontend