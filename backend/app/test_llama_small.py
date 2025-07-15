import os
from llama_cpp import Llama

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model directory
model_dir = os.path.join(current_dir, "..", "models")

# Construct the absolute path to the model file
model_path = os.path.join(model_dir, "qwen1_5-1_8b-chat-q8_0.gguf")  # Corrected model name

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit(1)

# Initialize the Llama model
try:
    llm = Llama(model_path=model_path, n_gpu_layers=40, n_ctx=512)  # Adjust n_gpu_layers and n_ctx
except Exception as e:
    print(f"Error initializing Llama model: {e}")
    exit(1)

# Perform a simple inference
try:
    output = llm("Question: What is the capital of France? Answer:", max_tokens=32, stop=["Q:", "\n"], echo=True)
    print(output)
    if "Paris" in output['choices'][0]['text']:
        print ("Test passed, the model works")
    else:
        print("Test failed, Paris not mentioned")
except Exception as e:
    print(f"Error during inference: {e}")
    exit(1)

print("Inference complete.")