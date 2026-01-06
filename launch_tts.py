#!/usr/bin/env python3
# launch_tts.py - Simple TTS service launcher

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Launch the TTS service"""
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent / "backend" / "app"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Set environment variables
    os.environ["TTS_PORT"] = "8002"
    os.environ["TTS_HOST"] = "0.0.0.0"
    
    # GPU optimization: Set performance environment variables
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    os.environ["CUDA_MEMORY_FRACTION"] = "0.9"
    
    # Use GPU 0 for TTS service (default)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE"] = "0"
    
    print("🚀 Starting TTS Service on port 8002...")
    print(f"📁 Backend directory: {backend_dir}")
    
    try:
        # Run the TTS backend
        uvicorn.run(
            "tts_backend:app",
            host="0.0.0.0",
            port=8002,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 TTS Service stopped by user")
    except Exception as e:
        print(f"❌ Failed to start TTS service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
