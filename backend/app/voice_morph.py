"""
Timbre morph for voice reference merging.

Blends voices in WORLD vocoder space (F0 + spectral envelope + aperiodicity),
similar in intent to celebrity face merges: both sources remain recognizable in
the hybrid rather than overlaying two speeches or replacing timbre via RVC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MORPH_SR = 44_100


def _load_mono(path: Path, sr: int = MORPH_SR) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32)


def _normalize_weights(weights: Sequence[float], n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("morph requires at least one voice")
    if weights:
        w = np.asarray(list(weights[:n]), dtype=np.float64)
        if w.size < n:
            pad = np.full(n - w.size, 1.0 / n, dtype=np.float64)
            w = np.concatenate([w, pad])
    else:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return w / total


def morph_voice_arrays(
    wavs: list[np.ndarray],
    *,
    sr: int = MORPH_SR,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if not wavs:
        raise ValueError("morph requires at least one waveform")
    if len(wavs) == 1:
        return np.asarray(wavs[0], dtype=np.float32)

    try:
        import pyworld as pw
    except ImportError as exc:
        raise RuntimeError(
            "pyworld is required for voice morph merging. Install with: pip install pyworld"
        ) from exc

    w = _normalize_weights(weights or [], len(wavs))
    n_samples = max(len(x) for x in wavs)
    padded = [
        np.pad(np.asarray(x, dtype=np.float64), (0, n_samples - len(x)))
        for x in wavs
    ]

    f0s: list[np.ndarray] = []
    sps: list[np.ndarray] = []
    aps: list[np.ndarray] = []
    frame_len: Optional[int] = None

    for wav in padded:
        _f0, times = pw.harvest(wav, sr)
        f0 = pw.stonemask(wav, _f0, times, sr)
        sp = pw.cheaptrick(wav, f0, times, sr)
        ap = pw.d4c(wav, f0, times, sr)
        if frame_len is None:
            frame_len = f0.shape[0]
        else:
            frame_len = min(frame_len, f0.shape[0], sp.shape[0], ap.shape[0])
        f0s.append(f0)
        sps.append(sp)
        aps.append(ap)

    assert frame_len is not None and frame_len > 0
    f0_m = sum(w[i] * f0s[i][:frame_len] for i in range(len(wavs)))
    sp_m = sum(w[i] * sps[i][:frame_len] for i in range(len(wavs)))
    ap_m = sum(w[i] * aps[i][:frame_len] for i in range(len(wavs)))

    out = pw.synthesize(f0_m, sp_m, ap_m, sr)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def morph_voice_files(
    paths: Sequence[Path],
    output_path: Path,
    *,
    sr: int = MORPH_SR,
    weights: Optional[Sequence[float]] = None,
) -> Path:
    if not paths:
        raise ValueError("morph requires at least one input file")
    wavs = [_load_mono(Path(p), sr=sr) for p in paths]
    merged = morph_voice_arrays(wavs, sr=sr, weights=weights)

    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), merged, sr, subtype="PCM_16")
    return output_path


def weights_from_balance(num_sources: int, balance: float) -> list[float]:
    """
    For two sources: balance 0 = all first voice, 1 = all second, 0.5 = equal morph.
    For N>2: balance is ignored; equal weights are used.
    """
    if num_sources <= 1:
        return [1.0]
    if num_sources == 2:
        b = float(np.clip(balance, 0.0, 1.0))
        return [1.0 - b, b]
    return [1.0 / num_sources] * num_sources
