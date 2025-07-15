import os
from pathlib import Path
import logging

# --- Configure these two paths to match your setup ---
# You can copy these paths from your Eloquent UI model settings
MODEL_PATH = "C:\\Users\\bpfit\\OneDrive\\Desktop\\LLM AI GGUFs\\gemma-3-27b-it-abliterated.q4_k_m.gguf"
MMPROJ_PATH = "C:\\Users\\bpfit\\OneDrive\\Desktop\\LLM AI GGUFs\\mmproj-mlabonne_gemma-3-27b-it-abliterated-f16.gguf" # Or whatever your mmproj file is named
# ----------------------------------------------------

# Basic setup to mimic your app's environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from llama_cpp import Llama
    logging.info("Successfully imported Llama.")
except ImportError as e:
    logging.error(f"Failed to import Llama. Your environment may not be set up correctly. Error: {e}")
    exit()

if not Path(MODEL_PATH).exists():
    logging.error(f"Model file not found at: {MODEL_PATH}")
    exit()

if not Path(MMPROJ_PATH).exists():
    logging.error(f"MMPROJ file not found at: {MMPROJ_PATH}")
    exit()

logging.info("Loading model with vision support for inspection...")

try:
    # We initialize the Llama object exactly as your app would, with the vision model
    model_instance = Llama(
        model_path=MODEL_PATH,
        clip_model_path=MMPROJ_PATH,
        n_gpu_layers=999, # Load fully on GPU
        verbose=False # Keep the output clean
    )

    logging.info("Model loaded successfully. Now inspecting the object...")
    print("\n" + "="*80)
    print("INSPECTION RESULTS")
    print("="*80 + "\n")

    # --- Test 1: Check for all methods containing 'chat' or 'completion' ---
    print("--- 1. Available Methods ---")
    print("Searching for methods related to 'chat' and 'completion':")
    found_methods = [method for method in dir(model_instance) if 'chat' in method or 'completion' in method]
    if found_methods:
        for method in found_methods:
            print(f"  - Found method: {method}")
    else:
        print("  - No methods containing 'chat' or 'completion' found.")
    print("\n" + "-"*80 + "\n")


    # --- Test 2: Get the detailed help documentation for the most likely chat method ---
    # We will try 'create_chat_completion' first as it's the most likely candidate
    method_to_inspect = 'create_chat_completion'
    if hasattr(model_instance, method_to_inspect):
        print(f"--- 2. Detailed Help for '{method_to_inspect}' ---")
        try:
            help(getattr(model_instance, method_to_inspect))
        except Exception as e:
            print(f"Could not get help for '{method_to_inspect}': {e}")
    else:
        print(f"--- 2. Detailed Help ---")
        print(f"Method '{method_to_inspect}' does not exist on the Llama object.")

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80 + "\n")


except Exception as e:
    logging.error(f"An error occurred during model loading or inspection: {e}", exc_info=True)