# launch.py - Fixed version

import os
import sys
import uvicorn
from multiprocessing import Process, freeze_support
import socket
import time
import webbrowser
import threading

def get_project_root():
    """Gets the absolute path to the project's root directory (where launch.py is)."""
    return os.path.dirname(os.path.abspath(__file__))


def get_gpu_count():
    """Dynamically detect available NVIDIA GPUs."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            gpu_count = len(gpus)
            if gpu_count > 0:
                print(f"Detected {gpu_count} NVIDIA GPU(s):")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")
                return gpu_count
    except Exception as e:
        print(f"GPU detection failed: {e}")
    
    # Fallback to PyTorch detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Detected {gpu_count} GPU(s) via PyTorch")
            return gpu_count
    except:
        pass
    
    print("No NVIDIA GPUs detected.")
    return 0


def launch_browser():
    """Launch browser after delay - runs in separate thread to avoid blocking startup"""
    try:
        time.sleep(25)  # Wait 25 seconds
        webbrowser.open('http://localhost:5173/')
        print("Browser launched at http://localhost:5173/")
    except Exception:
        pass  # Fail silently - don't break the app if browser launch fails


# MOVED TO MODULE LEVEL - Windows can't pickle local functions
def run_model_service(root_path):
    """Run the model service with proper GPU isolation."""
    gpu_count = get_gpu_count()
    if gpu_count >= 2:
        # If multiple GPUs, use the last GPU for main LLM inference
        model_gpu_id = str(gpu_count - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = model_gpu_id
        os.environ["GPU_ID"] = model_gpu_id
        print(f"🔒 Model service will use GPU {model_gpu_id} for main LLM inference")
    elif gpu_count == 1:
        # Single GPU - use it for everything
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["GPU_ID"] = "0"
        print("🔒 Model service will use GPU 0")
    else:
        # No GPU - run on CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["GPU_ID"] = "-1"
        print("🔒 Model service will run on CPU")
    
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    # Now run the actual service using subprocess without creating new console
    import subprocess
    service_script = os.path.join(root_path, "backend", "app", "model_service.py")
    subprocess.run([sys.executable, service_script], cwd=root_path)


def start_model_service(root_path):
    """Start the model service as a separate process."""
    # Check if already running
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect(('localhost', 5555))
        test_socket.close()
        print("Model service already running on port 5555")
        return None
    except:
        pass
    
    print("Starting model service on port 5555...")
    # Pass root_path as argument since it's at module level now
    service_process = Process(target=run_model_service, args=(root_path,))
    service_process.start()
    
    # Give it time to start (increased for better reliability)
    time.sleep(5)
    
    # Verify it started
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(10)  # Add timeout to prevent hanging
        test_socket.connect(('localhost', 5555))
        test_socket.close()
        print("✅ Model service started successfully")
        return service_process
    except Exception as e:
        print(f"❌ ERROR: Model service failed to start: {e}")
        print("🔄 Attempting to terminate and restart...")
        try:
            service_process.terminate()
            service_process.wait(timeout=5)
        except:
            service_process.kill()  # Force kill if terminate doesn't work
        return None


def start_backend(host, port, gpu_id, root_path):
    """Start backend with proper GPU assignment using subprocess with virtual environment"""
    try:
        import subprocess
        
        gpu_label = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        print(f"--- Starting Server Process for {gpu_label} on Port {port} ---")
        print(f"--- Using Python executable: {sys.executable} ---")
        
        # Build the command to run the backend
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info",
            "--ws-ping-interval", "300"
        ]
        
        # Set up environment variables for the subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPU_ID"] = str(gpu_id)
        env["PORT"] = str(port)
        env["PYTHONPATH"] = root_path
        
        print(f"--- Environment 'CUDA_VISIBLE_DEVICES' set to '{gpu_id}' ---")
        print(f"--- Command: {' '.join(cmd)} ---")
        
        # Launch the backend process (no new console - runs in same window)
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=root_path
        )
        
        print(f"✅ Backend process started (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to start backend server on GPU {gpu_id}, port {port}.", file=sys.stderr)
        print(e, file=sys.stderr)
        return None

def start_tts_service(root_path):
    """Start the TTS service in a new command window"""
    try:
        import subprocess
        
        # Build the command to run TTS service
        tts_script = os.path.join(root_path, "launch_tts.py")
        cmd = [sys.executable, tts_script]
        
        print("--- Starting TTS Service on Port 8002 ---")
        print(f"--- Command: {' '.join(cmd)} ---")
        
        # Launch TTS service (no new console - runs in same window)
        tts_process = subprocess.Popen(
            cmd,
            cwd=root_path
        )
        
        print(f"✅ TTS service launched (PID: {tts_process.pid})")
        return tts_process
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to start TTS service on port 8002.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)




def main():
    project_root = get_project_root()
    
    # Start browser launch timer in background thread
    browser_thread = threading.Thread(target=launch_browser, daemon=True)
    browser_thread.start()
    
    # Start the model service first
    model_service_process = start_model_service(project_root)
    
    gpu_count = get_gpu_count()
    processes = []
    
    if model_service_process:
        processes.append(model_service_process)
    
    if gpu_count == 0:
        print("No NVIDIA GPUs detected. Starting backend on CPU and TTS service.")
        main_backend = start_backend(host="0.0.0.0", port=8000, gpu_id=-1, root_path=project_root)
        if main_backend:
            processes.append(main_backend)
            print("✅ Main backend started on port 8000 (CPU)")
    elif gpu_count == 1:
        print(f"Found {gpu_count} NVIDIA GPU. Starting backend and TTS service on GPU 0.")
        main_backend = start_backend(host="0.0.0.0", port=8000, gpu_id=0, root_path=project_root)
        if main_backend:
            processes.append(main_backend)
            print("✅ Main backend started on port 8000 using GPU 0")
    else:
        print(f"Found {gpu_count} NVIDIA GPUs. Starting main backend, secondary backend, and TTS service.")
        
        # Start main backend on port 8000 using GPU 0
        main_backend = start_backend("0.0.0.0", 8000, 0, project_root)
        if main_backend:
            processes.append(main_backend)
            print("✅ Main backend started on port 8000 using GPU 0")
        
        # Start secondary backend on port 8001 using GPU 1 (if available)
        secondary_backend = start_backend("0.0.0.0", 8001, 1, project_root)
        if secondary_backend:
            processes.append(secondary_backend)
            print("✅ Secondary backend started on port 8001 using GPU 1")
    
    # ALWAYS start TTS service on port 8002 (regardless of GPU count)
    tts_service = start_tts_service(project_root)
    if tts_service:
        processes.append(tts_service)
        print("✅ TTS service launched on port 8002")
    
    # Wait for all processes
    if processes:
        try:
            for p in processes:
                if hasattr(p, 'join'):  # multiprocessing.Process
                    p.join()
                elif hasattr(p, 'wait'):  # subprocess.Popen
                    p.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")
            for p in processes:
                try:
                    if hasattr(p, 'terminate'):
                        p.terminate()
                except:
                    pass


if __name__ == "__main__":
    freeze_support()
    main()