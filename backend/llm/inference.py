import asyncio
import json
import logging
from typing import List, AsyncGenerator
import time
import os

from llama_cpp import Llama
from ctransformers import AutoModelForCausalLM

MODEL_DIR = r"C:\Users\bpfit\OneDrive\Desktop\LLM AI GGUFs"

class ModelManager:
    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_name: str, backend: str = "llama-cpp"):
        model_path = os.path.join(MODEL_DIR, model_name)
        if backend == "llama-cpp":
            model = Llama(model_path=model_path)
        elif backend == "ctransformers":
            model = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.loaded_models[model_name] = model

async def generate_text(
    model_manager: ModelManager,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: List[str] = [],
    backend: str = "llama-cpp"
) -> str:
    if model_name not in model_manager.loaded_models:
        model_manager.load_model(model_name, backend)

    model = model_manager.loaded_models[model_name]

    start_time = time.time()

    if backend == "llama-cpp":
        completion = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop_sequences or None
        )
        text = completion["choices"][0]["text"]

    elif backend == "ctransformers":
        text = model(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences or None
        )

    end_time = time.time()
    tokens_per_second = len(text.split()) / (end_time - start_time)
    logging.info(f"Generated {len(text.split())} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.2f} tokens/s)")

    return text


async def generate_text_stream(
    model_manager: ModelManager,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: List[str] = [],
    backend: str = "llama-cpp"
) -> AsyncGenerator[str, None]:

    if model_name not in model_manager.loaded_models:
        model_manager.load_model(model_name, backend)

    model = model_manager.loaded_models[model_name]

    start_time = time.time()
    token_count = 0

    if backend == "llama-cpp":
        completion_generator = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop_sequences or None,
            stream=True
        )

        for completion in completion_generator:
            token = completion["choices"][0]["text"]
            token_count += 1
            yield json.dumps({"text": token, "done": False})
            await asyncio.sleep(0.01)

    elif backend == "ctransformers":
        generated_text = ""
        for _ in range(max_tokens):
            next_token = model(
                prompt + generated_text,
                max_new_tokens=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop_sequences or None
            ).replace(prompt + generated_text, "")

            if not next_token or any(stop_seq in next_token for stop_seq in stop_sequences):
                break

            generated_text += next_token
            token_count += 1
            yield json.dumps({"text": next_token, "done": False})
            await asyncio.sleep(0.01)

    yield json.dumps({"text": "", "done": True})

    end_time = time.time()
    tokens_per_second = token_count / (end_time - start_time)
    logging.info(f"Streamed {token_count} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.2f} tokens/s)")