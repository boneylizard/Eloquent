import os
import struct
from pathlib import Path


_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_exact(handle, size):
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("The GGUF metadata is incomplete.")
    return value


def _read_string(handle):
    length = struct.unpack("<Q", _read_exact(handle, 8))[0]
    if length > 64 * 1024 * 1024:
        raise ValueError("The GGUF metadata contains an invalid string length.")
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _skip_value(handle, value_type):
    if value_type in _SCALAR_FORMATS:
        handle.seek(struct.calcsize(_SCALAR_FORMATS[value_type]), os.SEEK_CUR)
        return
    if value_type == 8:
        length = struct.unpack("<Q", _read_exact(handle, 8))[0]
        handle.seek(length, os.SEEK_CUR)
        return
    if value_type == 9:
        item_type = struct.unpack("<I", _read_exact(handle, 4))[0]
        item_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if item_type in _SCALAR_FORMATS:
            handle.seek(struct.calcsize(_SCALAR_FORMATS[item_type]) * item_count, os.SEEK_CUR)
            return
        for _ in range(item_count):
            _skip_value(handle, item_type)
        return
    raise ValueError(f"Unsupported GGUF metadata type: {value_type}")


def _read_value(handle, value_type):
    if value_type in _SCALAR_FORMATS:
        value_format = _SCALAR_FORMATS[value_type]
        return struct.unpack(value_format, _read_exact(handle, struct.calcsize(value_format)))[0]
    if value_type == 8:
        return _read_string(handle)
    _skip_value(handle, value_type)
    return None


def read_gguf_scalar_metadata(model_path):
    metadata = {}
    with Path(model_path).open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError("The selected file is not a GGUF model.")
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF version: {version}")
        _read_exact(handle, 8)
        metadata_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if metadata_count > 1_000_000:
            raise ValueError("The GGUF metadata count is invalid.")
        for _ in range(metadata_count):
            key = _read_string(handle)
            value_type = struct.unpack("<I", _read_exact(handle, 4))[0]
            value = _read_value(handle, value_type)
            if value is not None:
                metadata[key] = value
    return metadata


def estimate_gguf_memory(model_path, context_lengths=None):
    model_path = Path(model_path)
    metadata = read_gguf_scalar_metadata(model_path)
    architecture = metadata.get("general.architecture")
    if not architecture:
        raise ValueError("This GGUF does not declare its model architecture.")

    prefix = f"{architecture}."
    block_count = metadata.get(f"{prefix}block_count")
    embedding_length = metadata.get(f"{prefix}embedding_length")
    head_count = metadata.get(f"{prefix}attention.head_count")
    kv_head_count = metadata.get(f"{prefix}attention.head_count_kv", head_count)
    native_context = metadata.get(f"{prefix}context_length")

    required = (block_count, embedding_length, head_count, kv_head_count)
    if not all(isinstance(value, (int, float)) and value > 0 for value in required):
        raise ValueError("This GGUF does not include enough attention metadata for a context estimate.")

    kv_width = embedding_length * (kv_head_count / head_count)
    bytes_per_token = round(block_count * kv_width * 4)
    weights_bytes = model_path.stat().st_size
    runtime_allowance_bytes = max(round(weights_bytes * 0.05), 512 * 1024 * 1024)
    lengths = context_lengths or (4096, 8192, 16384, 32768)

    estimates = []
    for context_length in sorted({int(length) for length in lengths if int(length) > 0}):
        kv_cache_bytes = bytes_per_token * context_length
        estimates.append({
            "context_length": context_length,
            "kv_cache_bytes": kv_cache_bytes,
            "estimated_total_bytes": weights_bytes + runtime_allowance_bytes + kv_cache_bytes,
        })

    return {
        "model_name": model_path.name,
        "architecture": architecture,
        "weights_bytes": weights_bytes,
        "runtime_allowance_bytes": runtime_allowance_bytes,
        "kv_bytes_per_token": bytes_per_token,
        "native_context_length": int(native_context) if isinstance(native_context, (int, float)) else None,
        "estimates": estimates,
        "note": "Approximate working memory. GPU offload determines how it is divided between system RAM and VRAM.",
    }
