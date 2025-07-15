# model_subprocess.py
import multiprocessing as mp
from multiprocessing.connection import Listener, Client
import os
import logging
import queue
import time
import subprocess
import sys

def model_worker(model_path, gpu_id, model_params, address, ready_queue):
    """Worker function that runs in the subprocess with clean CUDA state"""
    # Set CUDA environment for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"
    
    # NOW import llama_cpp with clean CUDA state
    from llama_cpp import Llama
    
    try:
        # Load the model
        logging.info(f"Worker loading model on GPU {gpu_id}")
        model = Llama(model_path=model_path, **model_params)
        logging.info(f"Worker model loaded successfully")
        
        # Signal ready
        ready_queue.put("READY")
        
        # Set up IPC server
        listener = Listener(address, authkey=b'eloquent')
        
        while True:
            conn = listener.accept()
            try:
                while True:
                    msg = conn.recv()
                    if msg["type"] == "generate":
                        result = model(**msg["params"])
                        conn.send(result)
                    elif msg["type"] == "create_completion":
                        result = model.create_completion(**msg["params"])
                        conn.send(result)
                    elif msg["type"] == "shutdown":
                        return
            except EOFError:
                pass
            finally:
                conn.close()
                
    except Exception as e:
        import traceback
        ready_queue.put(f"ERROR: {str(e)}\n{traceback.format_exc()}")

class SubprocessModel:
    def __init__(self, model_name, model_path, gpu_id, model_params):
        self.model_name = model_name
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.model_params = model_params
        self.address = ('localhost', 6000 + gpu_id)
        self.process = None
        self.client = None
        
    def start(self):
        """Launch the model server as a separate Python process"""
        import json
        from pathlib import Path

        # Get the full path to model_server.py (same directory as this file)
        model_server_path = Path(__file__).parent / "model_server.py"

        # Launch the server script with FULL PATH
        cmd = [
            sys.executable,
            str(model_server_path),  # FULL PATH
            "--gpu", str(self.gpu_id),
            "--port", str(self.address[1]),
            "--model-path", self.model_path,
            "--model-params", json.dumps(self.model_params)
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to actually start and check for errors
        import time
        for i in range(30):  # 30 seconds for larger models
            time.sleep(1)

            # Check if process crashed
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise Exception(f"Model server crashed:\nSTDOUT: {stdout}\nSTDERR: {stderr}")

            # Read any output to see what's happening
            if self.process.stderr:
                try:
                    import select
                    # For Windows, we need a different approach
                    line = self.process.stderr.readline()
                    if line:
                        print(f"MODEL_SERVER: {line.strip()}")
                except:
                    pass

            # Try to connect
            try:
                self.client = Client(self.address, authkey=b'eloquent')
                logging.info("Connected to model server!")
                return  # Success!
            except ConnectionRefusedError:
                if i == 29:  # Last attempt
                    # Get any error output
                    self.process.terminate()
                    stdout, stderr = self.process.communicate()
                    raise Exception(f"Model server failed to start. Last output:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
                continue  # Not ready yet

        # Timeout
        self.process.terminate()
        raise Exception("Model server failed to start in 30 seconds")

    def __call__(self, *args, **kwargs):
        """Make it callable like the original Llama object"""
        self.client.send({"type": "generate", "params": kwargs})
        return self.client.recv()
        
    def create_completion(self, *args, **kwargs):
        """Support the create_completion method"""
        self.client.send({"type": "create_completion", "params": kwargs})
        return self.client.recv()
        
    def shutdown(self):
        """Clean shutdown"""
        if self.client:
            try:
                self.client.send({"type": "shutdown"})
                self.client.close()
            except:
                pass
        if self.process:
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()