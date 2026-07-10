from .voice_morph import morph_voice_arrays, weights_from_balance


def test_weights_from_balance_two_sources():
    assert weights_from_balance(2, 0.0) == [1.0, 0.0]
    assert weights_from_balance(2, 1.0) == [0.0, 1.0]
    assert weights_from_balance(2, 0.5) == [0.5, 0.5]
    w = weights_from_balance(2, 0.953)
    assert abs(w[0] - 0.047) < 1e-9
    assert abs(w[1] - 0.953) < 1e-9


def test_morph_single_passthrough():
    import numpy as np

    wav = np.zeros(8000, dtype=np.float32)
    out = morph_voice_arrays([wav])
    assert out.shape == wav.shape
