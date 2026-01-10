#!/usr/bin/env python3
# launch_tts.py - Simple TTS service launcher

import os
import sys
import uvicorn
from pathlib import Path

# Disable problematic Torch optimizations for Python 3.12+ (MUST BE AT TOP)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# MONKEYPATCH: Disable torch.compile to avoid Dynamo error on Python 3.12+
try:
    import torch
    if not hasattr(torch, '_original_compile'):
        torch._original_compile = torch.compile
        def dummy_compile(f, *args, **kwargs): return f
        torch.compile = dummy_compile
        print("🔧 [Monkeypatch] torch.compile disabled to bypass Dynamo 3.12+ error")
except Exception:
    pass

def main():
    """Launch the TTS service"""
    
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent / "backend" / "app"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # Read port from environment (set by launch.py) or use default
    tts_port = int(os.environ.get("TTS_PORT", 8002))
    tts_host = os.environ.get("TTS_HOST", "0.0.0.0")
    
    # GPU optimization: Set performance environment variables
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    os.environ["CUDA_MEMORY_FRACTION"] = "0.9"
    
    # Use GPU 0 for TTS service (default)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE"] = "0"
    
    print(f"🚀 Starting TTS Service on port {tts_port}...")
    print(f"📁 Backend directory: {backend_dir}")
    
    try:
        # Run the TTS backend
        uvicorn.run(
            "tts_backend:app",
            host=tts_host,
            port=tts_port,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 TTS Service stopped by user")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed to start TTS service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
