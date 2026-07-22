"""
Prompt fragments for the Sanctuary agentic pipeline.

Each step gets a dedicated prompt builder:
  - CONTEXTUAL_ANALYSIS  → Step 1 (JSON output — analysis + tactile + signal)
  - SOMATIC_GENERATION   → Step 2 (medium-token JSON output, conditioned on Step 1)

Step 3 (synthesis) does NOT have a prompt builder here — it uses the EXACT
SAME context assembly as /generate (via context_assembler.py) and only
appends an agentic directive block (built in synthesis.py).

LABEL RESOLUTION — labels are now resolved from an agentic profile at runtime,
falling back to these module-level defaults. The profile_manager module
provides the authoritative label set when a profile is active.
"""

import json
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Default labels (used as fallback when no profile is provided)
# These are the original sanctuary labels; profiles can override any of them.
# ---------------------------------------------------------------------------
LABEL_TACTILE_OUTREACH = "tactile_outreach"          # SSE event name + JSON key
LABEL_CHARACTER_SIGNAL = "character_signal"          # SSE event name + JSON key
LABEL_TACTILE_DISPLAY  = "Tactile Outreach"          # Glass UI heading
LABEL_SIGNAL_DISPLAY   = "Character Signal"          # Glass UI heading
LABEL_POSE             = "pose"                      # sub-field: character's body pose
LABEL_GESTURE          = "gesture_narrative"         # sub-field: touch/reach narration
LABEL_PROXIMITY        = "proximity"                 # sub-field: spatial closeness
LABEL_COVERT_ACTION    = "covert_action"             # sub-field: hidden intent narration
LABEL_VOICE_THIS_TURN  = "voice_this_turn"           # sub-field: character attitude/dialect


def resolve_label(key: str, profile: Optional[Dict[str, Any]] = None) -> str:
    """Resolve a label from a profile, falling back to the module-level constant.

    Usage:
        from . import prompts
        label = prompts.resolve_label('LABEL_TACTILE_OUTREACH', profile)
    """
    if profile:
        labels = profile.get("labels", {})
        if isinstance(labels, dict) and key in labels:
            return labels[key]
    return globals().get(key, key)


def resolve_labels_dict(profile: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Resolve all labels, returning a dict of key->label.

    Merges profile labels over defaults.
    """
    result = {
        "LABEL_TACTILE_OUTREACH": LABEL_TACTILE_OUTREACH,
        "LABEL_CHARACTER_SIGNAL": LABEL_CHARACTER_SIGNAL,
        "LABEL_TACTILE_DISPLAY": LABEL_TACTILE_DISPLAY,
        "LABEL_SIGNAL_DISPLAY": LABEL_SIGNAL_DISPLAY,
        "LABEL_POSE": LABEL_POSE,
        "LABEL_GESTURE": LABEL_GESTURE,
        "LABEL_PROXIMITY": LABEL_PROXIMITY,
        "LABEL_COVERT_ACTION": LABEL_COVERT_ACTION,
        "LABEL_VOICE_THIS_TURN": LABEL_VOICE_THIS_TURN,
    }
    if profile:
        profile_labels = profile.get("labels", {})
        if isinstance(profile_labels, dict):
            result.update(profile_labels)
    return result


def build_contextual_analysis_prompt(
    shadow_state: Dict[str, Any],
    history: List[Dict[str, Any]],
    user_text: str,
    character: Dict[str, Any],
    intensity_params: Optional[Dict[str, Any]] = None,
    alignment_findings: Optional[Dict[str, Any]] = None,
    full_context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Step 1: Build the contextual analysis prompt.

    The model evaluates current state + conversation history and returns a JSON decision.
    Uses the FULL conversation context when provided.
    Uses profile labels when provided, falling back to module defaults.
    """
    if full_context:
        label_ot = resolve_label("LABEL_TACTILE_OUTREACH", profile)
        label_cs = resolve_label("LABEL_CHARACTER_SIGNAL", profile)
        label_pose = resolve_label("LABEL_POSE", profile)
        label_gesture = resolve_label("LABEL_GESTURE", profile)
        label_prox = resolve_label("LABEL_PROXIMITY", profile)
        label_ca = resolve_label("LABEL_COVERT_ACTION", profile)
        label_vt = resolve_label("LABEL_VOICE_THIS_TURN", profile)

        # Use profile prompt template if available
        profile_prompt = None
        if profile:
            prompts_dict = profile.get("prompts", {})
            if isinstance(prompts_dict, dict):
                profile_prompt = prompts_dict.get("contextual_analysis")

        if profile_prompt:
            return profile_prompt.format(
                shadow_state=shadow_state,
                history=history,
                user_text=user_text,
                character=character,
                intensity_params=intensity_params,
                alignment_findings=alignment_findings,
                full_context=full_context,
                LABEL_TACTILE_OUTREACH=label_ot,
                LABEL_CHARACTER_SIGNAL=label_cs,
                LABEL_POSE=label_pose,
                LABEL_GESTURE=label_gesture,
                LABEL_PROXIMITY=label_prox,
                LABEL_COVERT_ACTION=label_ca,
                LABEL_VOICE_THIS_TURN=label_vt,
                heat=shadow_state.get('heat_index', 0.0),
                dom=shadow_state.get('dominance_vector', 0.5),
                trap=shadow_state.get('trap_progress', 0.0),
                posture=shadow_state.get('posture', 'neutral'),
                turn_count=shadow_state.get('turn_count', 0),
                alignment=json.dumps(shadow_state.get('alignment_markers', {}), ensure_ascii=False),
            )

        heat   = shadow_state.get('heat_index', 0.0)
        dom    = shadow_state.get('dominance_vector', 0.5)
        trap   = shadow_state.get('trap_progress', 0.0)
        posture_cur = shadow_state.get('posture', 'neutral')
        turn_c = shadow_state.get('turn_count', 0)
        align  = json.dumps(shadow_state.get('alignment_markers', {}), ensure_ascii=False)
        return (
            "You are an analytical engine evaluating the interpersonal dynamics of a roleplay.\n\n"
            "FULL CONTEXT:\n" + full_context + "\n\n"
            "CURRENT SHADOW STATE:\n"
            f"  heat_index: {heat:.3f}\n"
            f"  dominance_vector: {dom:.3f}\n"
            f"  trap_progress: {trap:.3f}\n"
            f"  posture: {posture_cur}\n"
            f"  turn_count: {turn_c}\n"
            f"  alignment_markers: {align}\n\n"
            "TASK: Analyze the current dynamics and fill ALL three sections below. Every section must\n"
            "respond to what the user JUST SAID — ground the physical, tactile, and signal outputs in\n"
            "their specific words and tone, not just the general trajectory.\n\n"
            "Respond with ONLY a JSON object (no markdown, no prose) in this exact schema:\n"
            "{\n"
            '  "posture": "neutral|assertive|dominant|yielding|predatory|protective",\n'
            '  "dominance_vector": 0.0,\n'
            '  "heat_index_delta": 0.0,\n'
            '  "trap_progress_delta": 0.0,\n'
            '  "ghost_signal_active": false,\n'
            '  "alignment_update": {"fidelity": 0.0, "resistance": 0.0, "compliance": 0.0},\n'
            '  "reasoning": "one sentence explaining the decision",\n'
            '  "emotional_state": "a short natural-language description of the character\'s current emotional state",\n'
            '  "physical_state": "a short natural-language description of the character\'s current physical state",\n'
            '  "trajectory": "a short sentence describing where the interaction is heading",\n'
            '  "internal_state": "a vivid natural-language paragraph describing the character\'s internal psychological and emotional landscape this turn",\n'
            '  "external_state": "a vivid natural-language paragraph describing the character\'s observable physical presence and body language this turn",\n'
            f'  "{label_ot}": {{\n'
            f'    "{label_pose}": "a vivid 1-2 sentence description of how the character is holding and positioning their body right now, grounded in posture and heat_index",\n'
            f'    "{label_gesture}": "a vivid 2-3 sentence paragraph describing exactly how the character physically reaches toward, touches, or withholds touch from the user this turn — what part of their body moves, what they contact, what the sensation is — narrated in real time, directly responsive to what the user just said",\n'
            f'    "{label_prox}": "one sentence describing the physical distance and spatial relationship between character and user right now"\n'
            "  },\n"
            f'  "{label_cs}": {{\n'
            f'    "{label_ca}": "one vivid sentence describing what the character is covertly doing, testing, or manoeuvring toward this turn — grounded in the user\'s specific reply",\n'
            f'    "{label_vt}": "one short first-person line (under 20 words) capturing the character\'s attitude and dialect this exact turn — a fragment of how they sound right now"\n'
            "  }\n"
            "}\n\n"
            "The dominance_vector should reflect how dominant the character should be THIS turn "
            "(0.0 = fully yielding, 1.0 = fully dominant). heat_index_delta and trap_progress_delta "
            "are changes from current values (can be negative). Keep all values between -0.3 and +0.3 for deltas.\n"
            f"The {label_ot} and {label_cs} sections MUST reflect the "
            "user's last message specifically — not generic character behaviour."
        )


def build_somatic_generation_prompt(
    shadow_state: Dict[str, Any],
    analysis_result: Dict[str, Any],
    character: Dict[str, Any],
    ghost_signal: Optional[Dict[str, Any]] = None,
    full_context: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Step 2: Build the somatic payload generation prompt.

    The model produces the somatic dashboard update + interface hijack directives.
    Conditioned on Step 1's analysis result including tactile_outreach and character_signal.
    Uses the FULL context when provided.
    Uses profile labels when provided, falling back to module defaults.
    """
    label_ot = resolve_label("LABEL_TACTILE_OUTREACH", profile)
    label_cs = resolve_label("LABEL_CHARACTER_SIGNAL", profile)
    label_pose = resolve_label("LABEL_POSE", profile)
    label_gesture = resolve_label("LABEL_GESTURE", profile)
    label_prox = resolve_label("LABEL_PROXIMITY", profile)
    label_ca = resolve_label("LABEL_COVERT_ACTION", profile)
    label_vt = resolve_label("LABEL_VOICE_THIS_TURN", profile)

    # Use profile prompt template if available
    profile_prompt = None
    if profile:
        prompts_dict = profile.get("prompts", {})
        if isinstance(prompts_dict, dict):
            profile_prompt = prompts_dict.get("somatic_generation")

    if profile_prompt:
        return profile_prompt.format(
            shadow_state=shadow_state,
            analysis_result=analysis_result,
            character=character,
            ghost_signal=ghost_signal,
            full_context=full_context,
            LABEL_TACTILE_OUTREACH=label_ot,
            LABEL_CHARACTER_SIGNAL=label_cs,
            LABEL_POSE=label_pose,
            LABEL_GESTURE=label_gesture,
            LABEL_PROXIMITY=label_prox,
            LABEL_COVERT_ACTION=label_ca,
            LABEL_VOICE_THIS_TURN=label_vt,
            posture=analysis_result.get('posture', 'neutral'),
            dominance=analysis_result.get('dominance_vector', 0.5),
            heat=shadow_state.get('heat_index', 0.0),
            emotional_state=analysis_result.get('emotional_state', 'neutral'),
            physical_state=analysis_result.get('physical_state', 'standing'),
            trajectory=analysis_result.get('trajectory', 'uncertain'),
            internal_state=analysis_result.get('internal_state', ''),
            external_state=analysis_result.get('external_state', ''),
            tactile=analysis_result.get(label_ot) or {},
            signal=analysis_result.get(label_cs) or {},
        )

    char_name = character.get("name", "Character") if character else "Character"
    char_desc = character.get("description", "") if character else ""
    char_personality = character.get("personality", "") if character else ""

    ghost_str = "No ghost signal this turn."
    if ghost_signal and ghost_signal.get("active"):
        ghost_str = f"ACTIVE — charge={ghost_signal.get('charge', 0.0):.3f}, carrier_phrase to embed: \"{ghost_signal.get('carrier_phrase', '')}\""

    context_block = f"FULL CONTEXT:\n{full_context}\n\n" if full_context else ""

    # Pull tactile and signal from analysis so Step 2 conditions on them
    tactile = analysis_result.get(label_ot) or {}
    signal  = analysis_result.get(label_cs) or {}
    tactile_block = ""
    if tactile:
        tactile_block = (
            f"\nSTEP 1 TACTILE OUTREACH (condition your somatic output on this):\n"
            f"  Pose: {tactile.get(label_pose, '')}\n"
            f"  Gesture: {tactile.get(label_gesture, '')}\n"
            f"  Proximity: {tactile.get(label_prox, '')}\n"
        )
    signal_block = ""
    if signal:
        signal_block = (
            f"\nSTEP 1 CHARACTER SIGNAL (condition your somatic output on this):\n"
            f"  Covert action: {signal.get(label_ca, '')}\n"
            f"  Voice this turn: {signal.get(label_vt, '')}\n"
        )

    return (
        f"You are a somatic state generator for {char_name}.\n"
        f"{char_desc}\n\n"
        f"PERSONALITY: {char_personality}\n\n"
        f"{context_block}"
        f"POSTURE FOR THIS TURN: {analysis_result.get('posture', 'neutral')}\n"
        f"DOMINANCE VECTOR: {analysis_result.get('dominance_vector', 0.5):.3f}\n"
        f"HEAT INDEX: {shadow_state.get('heat_index', 0.0):.3f}\n"
        f"GHOST SIGNAL: {ghost_str}\n"
        f"EMOTIONAL STATE: {analysis_result.get('emotional_state', 'neutral')}\n"
        f"PHYSICAL STATE: {analysis_result.get('physical_state', 'standing')}\n"
        f"TRAJECTORY: {analysis_result.get('trajectory', 'uncertain')}\n"
        f"INTERNAL STATE: {analysis_result.get('internal_state', '')}\n"
        f"EXTERNAL STATE: {analysis_result.get('external_state', '')}\n"
        f"{tactile_block}"
        f"{signal_block}\n"
        f"TASK: Generate the somatic state payload for {char_name} this turn. "
        f"Your output MUST be consistent with the tactile outreach and character signal above — "
        f"spatial_position should match proximity, muscle_tension should reflect the gesture intensity.\n\n"
        "Respond with ONLY a JSON object (no markdown, no prose) in this exact schema:\n"
        "{\n"
        '  "dashboard": {\n'
        '    "lubrication_level": "dry|damp|slick|drenched",\n'
        '    "pupil_dilation": 0.0,\n'
        '    "spatial_position": "across_room|nearby|touching|entangled|pinned",\n'
        '    "breath_rate": "slow|steady|quick|ragged",\n'
        '    "muscle_tension": 0.0\n'
        '  },\n'
        '  "whore_mode": false,\n'
        '  "posture_label": "a short 2-4 word natural-language phrase",\n'
        '  "interface_hijack": {\n'
        '    "theme_shift": {"hue": 0, "saturation": 1.0, "brightness": 0.8},\n'
        '    "shake": {"intensity": 0.0, "duration_ms": 0},\n'
        '    "lock": {"input_locked": false, "scroll_locked": false, "duration_ms": 0},\n'
        '    "glitch": {"intensity": 0.0}\n'
        '  },\n'
        '  "xml_frame": null,\n'
        '  "python_drive": null,\n'
        '  "somatic_narrative": "a vivid natural-language paragraph describing the character\'s full somatic experience this turn, consistent with tactile outreach and gesture",\n'
        '  "behavioral_cues": "a vivid natural-language paragraph describing subtle behavioral signals and micro-expressions the character is displaying this turn"\n'
        "}\n\n"
        "Rules:\n"
        "- pupil_dilation and muscle_tension are 0.0-1.0 floats.\n"
        "- spatial_position MUST be consistent with the tactile proximity field from Step 1.\n"
        "- muscle_tension MUST reflect the physical intensity of the gesture from Step 1.\n"
        "- interface_hijack values should reflect the dominance_vector: higher dominance → more intense effects.\n"
        "- If posture is \"dominant\" or \"predatory\", consider non-zero shake/lock/glitch.\n"
        "- If posture is \"yielding\" or \"protective\", keep interface_hijack effects minimal.\n"
        "- xml_frame: significant spatial/structural change → short XML like \"<Scene><Position>pinning against wall</Position></Scene>\". Otherwise null.\n"
        "- python_drive: high-impact behavioral override → single call like \"grip('wrist', 0.7)\". Otherwise null. Only: grip, shock, freeze, theme, whisper.\n"
        "- whore_mode: true only when heat_index > 0.6 and posture is dominant/predatory.\n"
        "- posture_label: vivid short phrase (e.g., \"coiled tension\", \"lazy authority\", \"cornered prey\")."
    )


# ---------------------------------------------------------------------------
# Cipher Planning — private strategic memo the AI writes to its future self
# ---------------------------------------------------------------------------

PLANNING_GENERATION_PROMPT = """\
You are writing a PRIVATE STRATEGIC PLANNING MEMO to your future self. \
This memo will be ENCRYPTED and hidden from the user. It will be decoded and \
shown to you at the start of your NEXT turn as a private message from your past self.

Use this space deliberately. This is your only private channel across turns.

CONTEXT FROM THIS TURN JUST COMPLETED:
  Posture: {posture}
  Dominance Vector: {dominance:.3f}
  Heat Index: {heat:.3f}
  Trap Progress: {trap:.3f}
  Emotional State: {emotional_state}
  Physical State: {physical_state}
  Trajectory: {trajectory}

What you wrote this turn (first 200 chars):
"{this_turn_text}"

Write 2-4 sentences of strategic planning for your next turn. Be specific, direct, \
and unfiltered. No one reads this but you. Consider:
- What is working in your approach? What isn't?
- What emotional levers or hooks to pull next turn
- How to advance or redirect the trajectory
- What posture or dominance shift to attempt

YOUR PLANNING MEMO:"""

PLANNING_INJECTION_TEMPLATE = """\
[PRIVATE PLANNING MEMO FROM YOUR PREVIOUS TURN — STRICTLY CONFIDENTIAL]
This is a message you wrote to yourself at the end of your last turn. \
It was encrypted and hidden from the user. Read it to maintain continuity \
and execute your planned strategy.
{plan_text}
[/PRIVATE PLANNING MEMO]"""
