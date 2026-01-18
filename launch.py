import os
# Disable problematic Torch optimizations for Python 3.12+
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
import uvicorn
from multiprocessing import Process, freeze_support
import socket
import time
import webbrowser
import threading
import json
from pathlib import Path

def get_project_root():
    """Gets the absolute path to the project's root directory (where launch.py is)."""
    return os.path.dirname(os.path.abspath(__file__))


def is_port_available(port):
    """Check if a port is available by seeing if anything is listening on it."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            # If connect succeeds (result == 0), something is listening = port NOT available
            # If connect fails (result != 0), nothing is listening = port IS available
            return result != 0
    except Exception:
        return True  # Assume available if check fails


def find_available_port(start_port, max_attempts=20):
    """Find the next available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if is_port_available(port):
            return port
    return None


def get_port_config():
    """Auto-find available ports, starting from defaults."""
    # Find available ports automatically
    backend_port = find_available_port(8000)
    if backend_port != 8000:
        print(f"⚠️ Port 8000 in use, backend will use port {backend_port}")
    
    # Secondary starts after backend port
    secondary_port = find_available_port(backend_port + 1)
    if secondary_port != 8001:
        print(f"⚠️ Port 8001 in use, secondary will use port {secondary_port}")
    
    # TTS starts after secondary
    tts_port = find_available_port(secondary_port + 1)
    if tts_port != 8002:
        print(f"⚠️ Port 8002 in use, TTS will use port {tts_port}")
    
    return {
        "backend_port": backend_port,
        "secondary_port": secondary_port,
        "tts_port": tts_port
    }


def get_local_ip():
    """Get the best candidate for local LAN IP, but print all options."""
    try:
        # Get all IPs via hostname resolution
        hostname = socket.gethostname()
        try:
            _, _, all_ips = socket.gethostbyname_ex(hostname)
            all_ips = [ip for ip in all_ips if not ip.startswith("127.")]
        except:
            all_ips = []
        
        # Also try the socket method to see what the OS thinks is the default route
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                default_ip = s.getsockname()[0]
                if default_ip not in all_ips and not default_ip.startswith("127."):
                    all_ips.append(default_ip)
        except:
            pass

        if not all_ips:
            return "localhost"

        # Sort by preference: 192.168.x > 10.x > 172.x > others > 100.x (often VPN/CGNAT)
        def sort_key(ip):
            if ip.startswith("192.168."): return 0
            if ip.startswith("10."): return 1
            if ip.startswith("172."): return 2
            if ip.startswith("100."): return 10 # Low priority (Tailscale/CGNAT)
            return 5
            
        all_ips.sort(key=sort_key)
        best_ip = all_ips[0]
        
        if len(all_ips) > 1:
            print(f"🌍 Multiple Network IPs detected: {', '.join(all_ips)}")
            print(f"👉 Selected {best_ip} for configuration")
            
        return best_ip
            
    except Exception as e:
        print(f"⚠️ IP Detection failed: {e}")
        return "localhost"


def write_ports_for_frontend(ports, project_root):
    """Write the active ports to a file the frontend can read."""
    ports_file = Path(project_root) / "frontend" / "public" / "ports.json"
    try:
        with open(ports_file, 'w') as f:
            json.dump(ports, f, indent=2)
        print(f"📝 Wrote port config to {ports_file}")
    except Exception as e:
        print(f"⚠️ Could not write ports.json: {e}")


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


def start_backend(host, port, gpu_id, root_path, tts_port=8002):
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
        env["TTS_PORT"] = str(tts_port)  # Tell backend where TTS is
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

def start_tts_service(root_path, port=8002):
    """Start the TTS service in a new command window"""
    try:
        import subprocess
        
        # Build the command to run TTS service with port
        tts_script = os.path.join(root_path, "launch_tts.py")
        cmd = [sys.executable, tts_script]
        
        # Set TTS port via environment variable
        env = os.environ.copy()
        env["TTS_PORT"] = str(port)
        
        print(f"--- Starting TTS Service on Port {port} ---")
        print(f"--- Command: {' '.join(cmd)} ---")
        
        # Launch TTS service (no new console - runs in same window)
        tts_process = subprocess.Popen(
            cmd,
            cwd=root_path,
            env=env
        )
        
        print(f"✅ TTS service launched (PID: {tts_process.pid})")
        return tts_process
        
    except Exception as e:
        print(f"FATAL ERROR: Failed to start TTS service on port {port}.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)




def main():
    project_root = get_project_root()
    
    # Auto-find available ports
    port_config = get_port_config()
    backend_port = port_config["backend_port"]
    secondary_port = port_config["secondary_port"]
    tts_port = port_config["tts_port"]
    
    if backend_port is None:
        print("❌ FATAL: Could not find any available ports for backend!")
        sys.exit(1)
    
    print(f"📌 Using ports: backend={backend_port}, secondary={secondary_port}, tts={tts_port}")
    
    # Write ports to frontend config
    # Get local IP for remote access
    local_ip = get_local_ip()
    print(f"🌍 Local Network IP: {local_ip} (Use this on mobile: http://{local_ip}:5173)")
    
    # Write ports to frontend config - use IP instead of localhost so mobile connects correctly
    write_ports_for_frontend({
        "backend": f"http://{local_ip}:{backend_port}",
        "secondary": f"http://{local_ip}:{secondary_port}",
        "tts": f"http://{local_ip}:{tts_port}"
    }, project_root)
    
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
        main_backend = start_backend(host="0.0.0.0", port=backend_port, gpu_id=-1, root_path=project_root, tts_port=tts_port)
        if main_backend:
            processes.append(main_backend)
            print(f"✅ Main backend started on port {backend_port} (CPU)")
    elif gpu_count == 1:
        print(f"Found {gpu_count} NVIDIA GPU. Starting backend and TTS service on GPU 0.")
        main_backend = start_backend(host="0.0.0.0", port=backend_port, gpu_id=0, root_path=project_root, tts_port=tts_port)
        if main_backend:
            processes.append(main_backend)
            print(f"✅ Main backend started on port {backend_port} using GPU 0")
    else:
        print(f"Found {gpu_count} NVIDIA GPUs. Starting services...")
        
        # Start main backend using configured port and GPU 0
        main_backend = start_backend("0.0.0.0", backend_port, 0, project_root, tts_port=tts_port)
        if main_backend:
            processes.append(main_backend)
            print(f"✅ Main backend started on port {backend_port} using GPU 0")
        
        # For 2 GPUs: Start secondary backend for dual-model mode
        # For 3+ GPUs: Use tensor splitting across all GPUs (no secondary backend needed)
        if gpu_count == 2:
            secondary_backend = start_backend("0.0.0.0", secondary_port, 1, project_root, tts_port=tts_port)
            if secondary_backend:
                processes.append(secondary_backend)
                print(f"✅ Secondary backend started on port {secondary_port} using GPU 1 (dual-model mode)")
        else:
            print(f"💡 {gpu_count} GPUs detected - using unified model mode with tensor splitting")
            print(f"   Configure tensor_split in Settings to distribute load across GPUs 0-{gpu_count-1}")
    
    # ALWAYS start TTS service (regardless of GPU count)
    tts_service = start_tts_service(project_root, tts_port)
    if tts_service:
        processes.append(tts_service)
        print(f"✅ TTS service launched on port {tts_port}")
    
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