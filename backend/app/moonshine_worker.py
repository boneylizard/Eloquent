#!/usr/bin/env python3
"""
Moonshine Streaming Tiny worker — runs in isolated venv with transformers >= 5.0.
Called by stt_service.py via subprocess.

Protocol:
  Input:  JSON line on stdin:  {"audio_path": "/path/to/audio.wav", "sample_rate": 16000}
  Output: JSON line on stdout: {"ok": true, "transcript": "..."} or {"ok": false, "error": "..."}

The model stays loaded across calls for fast subsequent transcriptions.
"""
import sys
import os
import json
import logging
import tempfile
import io
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("moonshine_worker")

MODEL_ID = "UsefulSensors/moonshine-streaming-tiny"

processor = None
model = None
device = None


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model():
    global processor, model, device
    if model is not None and processor is not None:
        return
    import torch
    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    device = get_device()
    logger.info(f"Loading Moonshine '{MODEL_ID}' on {device}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    logger.info(f"Moonshine loaded on {device}")


def transcribe(audio_path: str, target_sr: int = 16000) -> str:
    import torch
    import librosa
    import soundfile as sf
    import numpy as np

    load_model()

    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, res_type="kaiser_fast")
    except Exception:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
        sr = target_sr

    audio_duration = len(audio) / sr
    logger.info(f"Moonshine audio: {audio_duration:.2f}s")

    token_limit_factor = 6.5 / processor.feature_extractor.sampling_rate

    if audio_duration > 30:
        chunk_size = 20 * sr
        overlap = 2 * sr
        transcripts = []
        for i in range(0, len(audio), chunk_size - overlap):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < sr * 1:
                continue
            inputs = processor(chunk, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
            inputs = inputs.to(device)
            if device.startswith("cuda"):
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            seq_lens = inputs["attention_mask"].sum(dim=-1)
            max_length = int((seq_lens * token_limit_factor).max().item())
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=max_length)
            text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            if text:
                transcripts.append(text)
        return " ".join(transcripts)
    else:
        inputs = processor(audio, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
        inputs = inputs.to(device)
        if device.startswith("cuda"):
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        seq_lens = inputs["attention_mask"].sum(dim=-1)
        max_length = int((seq_lens * token_limit_factor).max().item())
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=max_length)
        return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def main_loop():
    logger.info("Moonshine worker ready, waiting for jobs on stdin...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            audio_path = payload["audio_path"]
            sr = payload.get("sample_rate", 16000)
            result = transcribe(audio_path, sr)
            output = json.dumps({"ok": True, "transcript": result})
        except Exception as e:
            logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
            output = json.dumps({"ok": False, "error": str(e)})
        sys.stdout.write(output + "\n")
        sys.stdout.flush()


def self_test():
    logger.info("Running self-test...")
    import torch
    import numpy as np
    load_model()
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        import soundfile as sf
        sf.write(path, audio, sr)
        result = transcribe(path, sr)
        logger.info(f"Self-test result: '{result}'")
        print(json.dumps({"ok": True, "transcript": result, "test": True}))
    finally:
        os.remove(path)


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        main_loop()
