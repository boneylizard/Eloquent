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
    """Safely checks for GPU count without initializing a full CUDA context."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        print(f"Detected {count} GPUs using pynvml.")
        return count
    except Exception:
        print("pynvml not found or failed, falling back to torch for GPU count.")
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
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
    """Run the model service - must be at module level for Windows"""
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    # Now run the actual service
    import subprocess
    service_script = os.path.join(root_path, "backend", "app", "model_service.py")
    subprocess.run([sys.executable, service_script])


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
    
    # Give it time to start
    time.sleep(3)
    
    # Verify it started
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.connect(('localhost', 5555))
        test_socket.close()
        print("Model service started successfully")
        return service_process
    except:
        print("ERROR: Model service failed to start")
        service_process.terminate()
        return None


def start_backend(host, port, gpu_id, root_path):
    """Your original function - unchanged"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["GPU_ID"] = str(gpu_id)
        os.environ["PORT"] = str(port)
        if root_path not in sys.path:
            sys.path.insert(0, root_path)
        print(f"--- Starting Server Process for GPU {gpu_id} on Port {port} ---")
        print(f"--- Environment 'CUDA_VISIBLE_DEVICES' set to '{os.environ['CUDA_VISIBLE_DEVICES']}' ---")
        print(f"--- Python path set to include: '{root_path}' ---")
        uvicorn.run("backend.app.main:app", host=host, port=port, log_level="info", reload=False, ws_ping_interval=300)
    except Exception as e:
        print(f"FATAL ERROR: Failed to start backend server on GPU {gpu_id}, port {port}.", file=sys.stderr)
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
    if gpu_count == 0:
        print("No NVIDIA GPUs detected. Starting one backend instance on CPU.")
        start_backend(host="0.0.0.0", port=8000, gpu_id=-1, root_path=project_root)
    else:
        print(f"Found {gpu_count} NVIDIA GPUs. Starting one server process per GPU.")
        processes = []
        
        if model_service_process:
            processes.append(model_service_process)
        
        for gpu_id in range(gpu_count):
            port = 8000 + gpu_id
            p = Process(target=start_backend, args=("0.0.0.0", port, gpu_id, project_root))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()


if __name__ == "__main__":
    freeze_support()
    main()