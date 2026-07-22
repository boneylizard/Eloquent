import struct

from .gguf_memory import estimate_gguf_memory


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def test_estimate_gguf_memory_uses_attention_metadata(tmp_path):
    metadata = {
        "general.architecture": (8, "llama"),
        "llama.block_count": (4, 32),
        "llama.embedding_length": (4, 4096),
        "llama.attention.head_count": (4, 32),
        "llama.attention.head_count_kv": (4, 8),
        "llama.context_length": (4, 32768),
    }
    payload = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, (value_type, value) in metadata.items():
        payload += _string(key) + struct.pack("<I", value_type)
        payload += _string(value) if value_type == 8 else struct.pack("<I", value)

    model_path = tmp_path / "test.gguf"
    model_path.write_bytes(payload + (b"x" * 1024))

    result = estimate_gguf_memory(model_path, [4096, 8192])

    assert result["architecture"] == "llama"
    assert result["native_context_length"] == 32768
    assert result["kv_bytes_per_token"] == 131072
    assert result["estimates"][1]["kv_cache_bytes"] == 1073741824
