"""
Sanctuary Evolution — Agentic Orchestration Framework.

Transforms the single-turn prompt-response chat pipeline into a three-step
agentic pipeline per user turn:

  Step 1 — Contextual Analysis     (contextual_analysis.py)
  Step 2 — Somatic Payload Gen     (somatic_generation.py)
  Step 3 — Natural Language Synth  (synthesis.py)

The pipeline is orchestrated by ``pipeline.py`` and exposed via ``router.py``
as ``POST /agentic/turn`` (multiplexed SSE) and ``GET /agentic/state/{user_id}/{character_id}``.

Profiles (configurable prompt templates + display labels) are managed by
``profile_manager.py`` and exposed via ``GET/POST/DELETE /agentic/profiles``.

Labels are now resolved from the active agentic profile at runtime, falling
back to the module-level constants in prompts.py when no profile is active.
The old ``sanctuary_patch.py`` CLI patching tool has been removed — use the
Agentic Profiles tab in the Settings UI instead.

All new code lives in this submodule; ``main.py`` receives exactly two lines
(import + include_router).
"""

from .router import router as sanctuary_router

__all__ = ["sanctuary_router"]
