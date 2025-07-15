# model_server.py - A standalone model server with clean CUDA state
import os
import sys
import argparse
import json

# Parse arguments BEFORE any CUDA imports
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--model-path', required=True)
parser.add_argument('--model-params', required=True)
args = parser.parse_args()

# Set CUDA environment BEFORE importing llama_cpp
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

# NOW import with clean state
from llama_cpp import Llama
from multiprocessing.connection import Listener
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - MODEL_SERVER - %(message)s')

# Load model
model_params = json.loads(args.model_params)
logging.info(f"Loading model on GPU {args.gpu}")
model = Llama(model_path=args.model_path, **model_params)
logging.info("Model loaded successfully with clean CUDA state!")

# Serve requests
listener = Listener(('localhost', args.port), authkey=b'eloquent')
logging.info(f"Listening on port {args.port}")

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
                sys.exit(0)
    except EOFError:
        pass
    finally:
        conn.close()