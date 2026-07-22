import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from backend.app import stt_service


def test_parakeet_cpp_failure_does_not_masquerade_as_whisper(monkeypatch):
    parakeet_error = RuntimeError("Parakeet.cpp is unavailable")
    parakeet = AsyncMock(side_effect=parakeet_error)
    whisper = AsyncMock(return_value="whisper fallback")
    monkeypatch.setattr(stt_service, "transcribe_with_parakeet_cpp_array", parakeet)
    monkeypatch.setattr(stt_service, "transcribe_with_whisper_array", whisper)

    with pytest.raises(RuntimeError, match="Parakeet.cpp is unavailable"):
        asyncio.run(
            stt_service._transcribe_audio_array(
                np.zeros(1600, dtype=np.float32),
                16000,
                "parakeet-cpp:tdt_ctc-110m:f16",
            )
        )

    whisper.assert_not_awaited()
