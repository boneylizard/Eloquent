"""
Ghost Signal injection.

Implements "Ghost Signals" (AGENTS.md §4): injecting high-charge variables
into ostensibly neutral technical discourse. The ghost signal is a hidden
somatic channel — the carrier phrase appears in the text as a normal
sentence, but the somatic payload carries the actual charge variable.

This module decides whether to activate a ghost signal for the current turn
and selects the carrier phrase pattern.
"""

import logging
import random
from typing import Any, Dict, Optional

logger = logging.getLogger("sanctuary.ghost_signal")

_CARRIER_TEMPLATES = [
    "The architecture requires careful consideration of load distribution.",
    "System calibration is proceeding within expected parameters.",
    "The lattice structure shows remarkable tensile integrity.",
    "Thermal regulation is maintaining optimal operating conditions.",
    "The signal-to-noise ratio has shifted noticeably in the last cycle.",
    "Calibration drift is within acceptable tolerance bands.",
    "The feedback loop is converging on a stable attractor.",
    "Resonance patterns suggest a natural harmonic alignment.",
]


def maybe_inject(
    ghost_signal_active: bool,
    heat_index: float,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """Determine if a ghost signal should be injected this turn.

    Args:
        ghost_signal_active: Whether the analysis step flagged ghost signaling.
        heat_index: Current heat index (0.0-1.0). Higher heat → higher charge.
        rng: Optional Random for deterministic testing.

    Returns:
        A ghost signal dict, or None if no injection this turn.
        The dict has: {active, charge, carrier_phrase}
    """
    if not ghost_signal_active:
        return None

    r = rng or random
    carrier = r.choice(_CARRIER_TEMPLATES)
    charge = max(0.0, min(1.0, heat_index * r.uniform(0.6, 1.0)))

    logger.debug(
        "ghost_signal.maybe_inject: active=%s charge=%.3f carrier=%r",
        ghost_signal_active, charge, carrier[:50],
    )

    return {
        "active": True,
        "charge": round(charge, 4),
        "carrier_phrase": carrier,
    }
