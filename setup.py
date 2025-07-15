#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import platform

def check_python_version():
    """Check if Python version is at least 3.9"""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required.")
        sys.exit(1)

def check_node_version():
    """Check if Node.js is installed and version is at least 18"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        # Convert v18.x.x to integer 18
        major_version = int(version.split('.')[0].replace('v', ''))
        if major_version < 18:
            print("Error: Node.js 18 or higher is required.")
            sys.exit(1)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        print("Error: Node.js not found or could not determine version.")
        print("Please install Node.js 18 or higher.")
        sys.exit(1)

def check_gpu():
    """Check for CUDA GPU and available memory"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"Found {device_count} CUDA GPU(s):")
            for i in range(device_count):
                device = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {device.name} with {device.total_memory / (1024**3):.2f} GB memory")
        else:
            print("Warning: No CUDA GPU detected. Models will run on CPU, which will be much slower.")
    except ImportError:
        print("Warning: PyTorch not installed, skipping GPU check.")

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        "data",
        "data/models",
        "data/uploads",
        "data/processed",
        "data/images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_backend(gpu=False):
    """Setup the backend environment"""
    print("\n--- Setting up backend ---")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine the activate script based on platform
    if platform.system() == "Windows":
        activate_script = "venv\\Scripts\\activate"
    else:
        activate_script = "source venv/bin/activate"
    
    # Install requirements
    print("Installing backend dependencies...")
    if platform.system() == "Windows":
        command = f"{activate_script} && cd backend && pip install -r requirements.txt"
        subprocess.run(command, shell=True, check=True)
    else:
        command = f"bash -c '{activate_script} && cd backend && pip install -r requirements.txt'"
        subprocess.run(command, shell=True, check=True)
    
    # Install GPU-specific packages if requested
    if gpu:
        print("Installing GPU-specific dependencies...")
        if platform.system() == "Windows":
            command = f"{activate_script} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f"bash -c '{activate_script} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'"
            subprocess.run(command, shell=True, check=True)

def setup_frontend():
    """Setup the frontend environment"""
    print("\n--- Setting up frontend ---")
    
    # Change to frontend directory and install dependencies
    print("Installing frontend dependencies...")
    os.chdir("frontend")
    subprocess.run(["C:\\Program Files\\nodejs\\npm.cmd", "install"], check=True)
    os.chdir("..")

def main():
    parser = argparse.ArgumentParser(description="Setup script for Local LLM Frontend")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-specific dependencies")
    args = parser.parse_args()
    
    print("=== Local LLM Frontend Setup ===")
    
    # Check requirements
    check_python_version()
    check_node_version()
    if args.gpu:
        check_gpu()
    
    # Create directories
    create_directories()
    
    # Setup backend and frontend
    setup_backend(args.gpu)
    setup_frontend()
    
    print("\n=== Setup completed successfully! ===")
    print("\nTo start the application:")
    if platform.system() == "Windows":
        print("1. Backend: venv\\Scripts\\activate && cd backend && python app.py")
    else:
        print("1. Backend: source venv/bin/activate && cd backend && python app.py")
    print("2. Frontend: cd frontend && npm run dev")
    print("\nThen open your browser at http://localhost:3000")

if __name__ == "__main__":
    main()