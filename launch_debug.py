# launch_debug.py - Debug version that keeps windows open on crash

import os
import sys
import uvicorn
from multiprocessing import Process, freeze_support
import socket
import time
import webbrowser
import threading
import subprocess

def get_project_root():
    """Gets the absolute path to the project's root directory (where launch.py is)."""
    return os.path.dirname(os.path.abspath(__file__))

def get_gpu_count():
    """Dynamically detect available NVIDIA GPUs."""
    try:
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

def run_model_service(root_path):
    """Run the model service with proper GPU isolation."""
    try:
        gpu_count = get_gpu_count()
        if gpu_count >= 2:
            model_gpu_id = str(gpu_count - 1)
            os.environ["CUDA_VISIBLE_DEVICES"] = model_gpu_id
            os.environ["GPU_ID"] = model_gpu_id
            print(f"Model service will use GPU {model_gpu_id} for main LLM inference")
        elif gpu_count == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["GPU_ID"] = "0"
            print("Model service will use GPU 0")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["GPU_ID"] = "-1"
            print("Model service will run on CPU")
        
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        
        # Now run the actual service
        service_script = os.path.join(root_path, "backend", "app", "model_service.py")
        subprocess.run([sys.executable, service_script])
    except Exception as e:
        print(f"ERROR: Model service crashed: {e}")
        input("Press Enter to close this window...")

def start_model_service(root_path):
    """Start the model service as a separate process."""
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect(('localhost', 5555))
        test_socket.close()
        print("Model service already running on port 5555")
        return None
    except:
        pass
    
    print("Starting model service on port 5555...")
    service_process = Process(target=run_model_service, args=(root_path,))
    service_process.start()
    
    time.sleep(5)
    
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(10)
        test_socket.connect(('localhost', 5555))
        test_socket.close()
        print("SUCCESS: Model service started successfully")
        return service_process
    except Exception as e:
        print(f"ERROR: Model service failed to start: {e}")
        try:
            service_process.terminate()
            service_process.wait(timeout=5)
        except:
            service_process.kill()
        return None

def start_backend_debug(host, port, gpu_id, root_path):
    """Start backend with debug mode that keeps window open on crash"""
    try:
        gpu_label = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        print(f"--- Starting Server Process for {gpu_label} on Port {port} ---")
        
        # Create a batch file that will keep the window open
        batch_content = f'''@echo off
echo Starting backend on port {port} with GPU {gpu_id}...
cd /d "{root_path}"
set CUDA_VISIBLE_DEVICES={gpu_id}
set GPU_ID={gpu_id}
set PORT={port}
set PYTHONPATH={root_path}
python -m uvicorn backend.app.main:app --host {host} --port {port} --log-level info --ws-ping-interval 300
if errorlevel 1 (
    echo.
    echo ERROR: Backend crashed with error code %errorlevel%
    echo.
    pause
) else (
    echo.
    echo SUCCESS: Backend stopped normally
    echo.
    pause
)
'''
        
        batch_file = os.path.join(root_path, f"start_backend_{port}.bat")
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        # Launch the batch file in a new console
        process = subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k", batch_file],
            cwd=root_path
        )
        
        print(f"SUCCESS: Backend process started (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to start backend server on GPU {gpu_id}, port {port}.")
        print(e)
        return None

def start_tts_service_debug(root_path):
    """Start the TTS service in debug mode that keeps window open on crash"""
    try:
        # Create a batch file for TTS service
        batch_content = f'''@echo off
echo Starting TTS Service on port 8002...
cd /d "{root_path}"
set TTS_PORT=8002
set TTS_HOST=0.0.0.0
set CUDA_VISIBLE_DEVICES=1
set CUDA_DEVICE=0
python launch_tts.py
if errorlevel 1 (
    echo.
    echo ERROR: TTS Service crashed with error code %errorlevel%
    echo.
    pause
) else (
    echo.
    echo SUCCESS: TTS Service stopped normally
    echo.
    pause
)
'''
        
        batch_file = os.path.join(root_path, "start_tts.bat")
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        # Launch the batch file in a new console
        tts_process = subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k", batch_file],
            cwd=root_path
        )
        
        print(f"SUCCESS: TTS service launched in new window (PID: {tts_process.pid})")
        return tts_process
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to start TTS service on port 8002.")
        print(e)
        return None

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
        main_backend = start_backend_debug(host="0.0.0.0", port=8000, gpu_id=-1, root_path=project_root)
        if main_backend:
            processes.append(main_backend)
            print("SUCCESS: Main backend started on port 8000 (CPU)")
    elif gpu_count == 1:
        print(f"Found {gpu_count} NVIDIA GPU. Starting backend and TTS service on GPU 0.")
        main_backend = start_backend_debug(host="0.0.0.0", port=8000, gpu_id=0, root_path=project_root)
        if main_backend:
            processes.append(main_backend)
            print("SUCCESS: Main backend started on port 8000 using GPU 0")
    else:
        print(f"Found {gpu_count} NVIDIA GPUs. Starting main backend, secondary backend, and TTS service.")
        
        # Start main backend on port 8000 using GPU 0
        main_backend = start_backend_debug("0.0.0.0", 8000, 0, project_root)
        if main_backend:
            processes.append(main_backend)
            print("SUCCESS: Main backend started on port 8000 using GPU 0")
        
        # Start secondary backend on port 8001 using GPU 1
        secondary_backend = start_backend_debug("0.0.0.0", 8001, 1, project_root)
        if secondary_backend:
            processes.append(secondary_backend)
            print("SUCCESS: Secondary backend started on port 8001 using GPU 1")
    
    # ALWAYS start TTS service on port 8002 (regardless of GPU count)
    tts_service = start_tts_service_debug(project_root)
    if tts_service:
        processes.append(tts_service)
        print("SUCCESS: TTS service launched on port 8002")
    
    # Wait for all processes
    if processes:
        try:
            for p in processes:
                if hasattr(p, 'join'):
                    p.join()
                elif hasattr(p, 'wait'):
                    p.wait()
        except KeyboardInterrupt:
            print("\nShutting down services...")
            for p in processes:
                try:
                    if hasattr(p, 'terminate'):
                        p.terminate()
                except:
                    pass

if __name__ == "__main__":
    freeze_support()
    main()
