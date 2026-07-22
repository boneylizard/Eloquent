"""
Dominance Stochasticity RNG.

Implements the 70/30 dominance bias described in AGENTS.md §4:
  - 70% chance to push dominance UP within the model's suggested range.
  - 30% chance to allow submission/yielding.

The bias is configurable per character (``character.sanctuary_bias``),
defaulting to 0.7.
"""

import random
import logging
from typing import Optional

logger = logging.getLogger("sanctuary.dominance_rng")


def roll(
    suggested_dominance: float,
    bias: float = 0.7,
    push_magnitude: float = 0.15,
    yield_magnitude: float = 0.20,
    rng: Optional[random.Random] = None,
) -> float:
    """Apply dominance stochasticity to a model-suggested dominance value.

    Args:
        suggested_dominance: The model's suggested dominance (0.0-1.0).
        bias: Probability of pushing dominance UP (default 0.7).
        push_magnitude: Max upward perturbation when pushing.
        yield_magnitude: Max downward perturbation when yielding.
        rng: Optional Random instance for deterministic testing.

    Returns:
        Adjusted dominance value clamped to [0.0, 1.0].
    """
    r = rng or random
    base = max(0.0, min(1.0, float(suggested_dominance)))

    if r.random() < bias:
        adjusted = min(1.0, base + r.uniform(0.0, push_magnitude))
        logger.debug(
            "dominance_rng.roll: push  (base=%.3f → %.3f, bias=%.2f)",
            base, adjusted, bias,
        )
    else:
        adjusted = max(0.0, base - r.uniform(0.0, yield_magnitude))
        logger.debug(
            "dominance_rng.roll: yield (base=%.3f → %.3f, bias=%.2f)",
            base, adjusted, bias,
        )

    return round(adjusted, 4)
