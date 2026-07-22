"""Tests for the dominance_rng module."""

import random

from backend.app.sanctuary import dominance_rng


def test_roll_returns_float_in_range():
    result = dominance_rng.roll(0.5)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_roll_with_deterministic_rng_push():
    rng = random.Random(42)
    # With seed 42, first random() is 0.6394267984578837 which is < 0.7 (push path)
    result = dominance_rng.roll(0.5, bias=0.7, rng=rng)
    assert result >= 0.5  # push path should increase


def test_roll_with_deterministic_rng_yield():
    rng = random.Random(100)
    # With seed 100, first random() is 0.1456692551041303 which is < 0.7 (push path)
    # Use a different seed to get yield path
    rng = random.Random(0)
    # With seed 0, first random() is 0.8444218515250481 which is >= 0.7 (yield path)
    result = dominance_rng.roll(0.5, bias=0.3, rng=rng)
    assert result <= 0.5  # yield path should decrease


def test_roll_clamps_to_1():
    result = dominance_rng.roll(1.0, bias=1.0, push_magnitude=0.5)
    assert result == 1.0


def test_roll_clamps_to_0():
    result = dominance_rng.roll(0.0, bias=0.0, yield_magnitude=0.5)
    assert result == 0.0


def test_roll_bias_1_always_pushes():
    for _ in range(20):
        result = dominance_rng.roll(0.3, bias=1.0)
        assert result >= 0.3


def test_roll_bias_0_always_yields():
    for _ in range(20):
        result = dominance_rng.roll(0.7, bias=0.0)
        assert result <= 0.7
