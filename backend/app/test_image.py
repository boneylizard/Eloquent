import json
import base64
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

def test():
    # Paths
    model_path = Path("C:/Users/bpfit/OneDrive/Desktop/LLM AI GGUFs/gemma-3-4b-it-Q6_K.gguf")
    mmproj_path = Path("C:/Users/bpfit/OneDrive/Desktop/LLM AI GGUFs/mmproj-F32.gguf")
    test_image = Path("C:/Users/bpfit/Downloads/file_00000000684861f889454b35a506307f.png")

    if not mmproj_path.exists():
        print(f"ERROR: mmproj file not found at {mmproj_path}")
        return

    if not test_image.exists():
        print(f"ERROR: test image not found at {test_image}")
        return

    print(f"Loading model: {model_path.name}")
    print(f"Using mmproj: {mmproj_path.name}")

    # Set up Llava15 vision chat handler
    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))

    # Load vision-capable model with handler
    model = Llama(
        model_path=str(model_path),
        chat_handler=chat_handler,
        n_ctx=8192,
        n_gpu_layers=-1,
        logits_all=True
    )

    # Load and encode image
    with open(test_image, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    }]

    print(f"\nMessage length: {len(json.dumps(messages))}")
    print("\n=== TESTING VISION ===")

    # Send to model
    response = model.create_chat_completion(messages=messages, max_tokens=100)
    print(f"\nVision response: {response['choices'][0]['message']['content']}")

if __name__ == "__main__":
    test()
