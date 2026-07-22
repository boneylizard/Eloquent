"""
Context Assembler — extracted from /generate in main.py.

This is the EXACT same context assembly logic that the /generate endpoint uses,
extracted into a reusable function so the Sanctuary agentic pipeline can call it
without duplicating the logic.

It takes the same request body fields that /generate receives and produces
the same fully-assembled `llm_prompt` string that gets sent to the LLM.

Steps replicated from main.py /generate:
  Step 3: Split prompt into character_persona + user_query
  Step 5: Legacy memory fetch (skipped when directProfileInjection is on)
  Step 6: Build interaction_components (timestamp, memory, summary, RAG, author note, intensity, reinforcement, user query)
  Step 9: Assemble final llm_prompt with [SYSTEM TRUTH] wrapper
"""

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .prompts import PLANNING_INJECTION_TEMPLATE

logger = logging.getLogger("sanctuary.context_assembler")


def split_prompt(original_client_prompt: str) -> Tuple[str, str]:
    """Step 3: Split the formatted prompt into persona+history and user query.

    Replicates the exact logic from main.py lines 5365-5434.
    Returns (character_persona_from_split, user_query_from_split).
    """
    character_persona_from_split = ""
    user_query_from_split = ""

    if "<|im_start|>user" in original_client_prompt:
        last_user_turn_start = original_client_prompt.rfind("<|im_start|>user")
        character_persona_from_split = original_client_prompt[:last_user_turn_start].strip()
        temp_query_block = original_client_prompt[last_user_turn_start:]
        user_content_match = re.search(r"<\|im_start\|>user\s*\n(.*?)(?:<\|im_end\|>|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            user_query_from_split = temp_query_block.replace("<|im_start|>user", "").strip()

    elif "[INST]" in original_client_prompt:
        last_inst_start = original_client_prompt.rfind("[INST]")
        character_persona_from_split = original_client_prompt[:last_inst_start].strip()
        temp_query_block = original_client_prompt[last_inst_start:]
        user_content_match = re.search(r"\[INST\](.*?)(?:\[/INST\]|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            user_query_from_split = temp_query_block.replace("[INST]", "").strip()
        user_query_from_split = re.sub(r"<<SYS>>.*?<</SYS>>", "", user_query_from_split, flags=re.DOTALL).strip()

    elif "<start_of_turn>user" in original_client_prompt:
        last_user_turn_start = original_client_prompt.rfind("<start_of_turn>user")
        character_persona_from_split = original_client_prompt[:last_user_turn_start].strip()
        temp_query_block = original_client_prompt[last_user_turn_start:]
        user_content_match = re.search(r"<start_of_turn>user\n(.*?)(?:<end_of_turn>|$)", temp_query_block, re.DOTALL)
        if user_content_match:
            user_query_from_split = user_content_match.group(1).strip()
        else:
            user_query_from_split = temp_query_block.replace("<start_of_turn>user\n", "").strip()

    elif "Human:" in original_client_prompt:
        parts = original_client_prompt.rsplit("Human:", 1)
        character_persona_from_split = parts[0].strip()
        user_query_from_split = parts[1].strip()

    elif "User Query:" in original_client_prompt:
        parts = original_client_prompt.split("User Query:", 1)
        character_persona_from_split = parts[0].strip()
        user_query_from_split = parts[1].strip()

    elif "User:" in original_client_prompt:
        parts = original_client_prompt.rsplit("User:", 1)
        character_persona_from_split = parts[0].strip()
        user_query_from_split = parts[1].strip()

    else:
        user_query_from_split = original_client_prompt.strip()
        character_persona_from_split = ""

    return character_persona_from_split, user_query_from_split


async def fetch_legacy_memory(
    user_id: str,
    user_profile: Dict[str, Any],
    user_query: str,
    active_character: Optional[Dict[str, Any]],
    memory_port: int,
    direct_profile_injection: bool,
    is_api: bool,
    request_purpose: Optional[str],
) -> str:
    """Step 5: Fetch legacy memory context (skipped when directProfileInjection is on).

    Replicates the logic from main.py lines 5453-5497.
    Returns memory_context_for_llm string (empty if skipped).
    """
    skip_legacy_memory = (
        direct_profile_injection
        or is_api
        or request_purpose in ["title_generation", "model_judging", "model_testing", "continuation",
                                "call_mode_character_about", "character_intro", "system_intro"]
    )

    if skip_legacy_memory:
        logger.info("context_assembler: skipping legacy memory fetch (directProfileInjection=%s, api=%s)",
                    direct_profile_injection, is_api)
        return ""

    if not user_id:
        logger.info("context_assembler: skipping legacy memory fetch (no user_id)")
        return ""

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:{memory_port}/memory/relevant",
                json={
                    "prompt": user_query[:300],
                    "userProfile": user_profile,
                    "systemTime": datetime.datetime.now().isoformat(),
                    "requestType": "generate_user_chat",
                    "active_character": active_character,
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            data = resp.json()
            memory_context = data.get("formatted_memories", "")
            if memory_context:
                logger.info("context_assembler: retrieved %d memories, %d chars",
                            data.get("memory_count", 0), len(memory_context))
            return memory_context
    except Exception as e:
        logger.error("context_assembler: memory context fetch error: %s", e, exc_info=True)
        return ""


def build_interaction_components(
    user_query_from_split: str,
    inject_timestamp: bool,
    memory_context_for_llm: str,
    summary_context: Optional[str],
    use_rag: bool,
    rag_docs: List[str],
    rag_available: bool,
    author_note: Optional[str],
    anti_repetition_mode: bool,
    intensity_params: Optional[Dict[str, Any]],
    user_profile_reinforcement: Optional[str],
    request_purpose: Optional[str],
    previous_plan_text: Optional[str] = None,
) -> str:
    """Step 6: Build the interaction_components block.

    Replicates the logic from main.py lines 5504-5957.
    Returns the final_interaction_block string.
    """
    interaction_components: List[str] = []

    # Timestamp
    if inject_timestamp and request_purpose not in ["title_generation", "model_testing", "model_judging"]:
        ts_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        interaction_components.append(f"[Current date and time: {ts_str}]")

    # Legacy memory
    if memory_context_for_llm:
        interaction_components.append("RELEVANT USER INFORMATION:\n" + memory_context_for_llm)

    # Summary context
    if summary_context and request_purpose not in [
        "title_generation", "model_judging", "model_testing", "continuation",
        "call_mode_character_about", "character_intro", "system_intro",
    ]:
        interaction_components.append("[PREVIOUS STORY SUMMARY]:\n" + summary_context + "\n[End of Summary]")

    # RAG
    if use_rag and request_purpose != "title_generation" and rag_available:
        try:
            from .. import rag_utils
            rag_res = rag_utils.query_documents(
                question=user_query_from_split,
                doc_ids=rag_docs or [],
                top_k=rag_utils.RAG_CHAT_TOP_K,
                threshold=rag_utils.RAG_CHAT_SIMILARITY_THRESHOLD,
            )
            if rag_res.get("status") == "success":
                rag_content = rag_res.get("formatted_context", "")
                if rag_content:
                    interaction_components.append("DOCUMENT CONTEXT:\n" + rag_content)
                    logger.info("context_assembler: added %d RAG chunks", len(rag_res.get("chunks", [])))
        except Exception as e:
            logger.error("context_assembler: RAG error: %s", e, exc_info=True)

    # Author's Note
    if author_note and author_note.strip() and request_purpose not in [
        "title_generation", "model_testing", "model_judging", "continuation",
    ]:
        interaction_components.append(f"[AUTHOR'S NOTE - Writing style guidance for this response]\n{author_note.strip()}")

    # Anti-repetition
    if anti_repetition_mode and request_purpose not in [
        "title_generation", "model_testing", "model_judging", "continuation",
        "book_chapter_json_outline",
    ]:
        interaction_components.append(
            "[VARIETY GUIDANCE]\n"
            "Each response should feel fresh and unique. Avoid:\n"
            "- Reusing paragraph structures or openings from your previous messages\n"
            "- Repeating descriptive phrases you've already used in this conversation\n"
            "- Formulaic greeting or closing patterns\n"
            "Vary your sentence structure and word choices naturally."
        )

    # Intensity guidance
    if intensity_params and request_purpose not in [
        "title_generation", "model_testing", "model_judging", "continuation",
        "book_chapter_json_outline",
    ]:
        intensity_block = _build_intensity_block(intensity_params)
        if intensity_block:
            interaction_components.append(intensity_block)

    # Profile reinforcement
    if user_profile_reinforcement and user_profile_reinforcement.strip() and request_purpose not in [
        "title_generation", "model_testing", "model_judging", "continuation",
    ]:
        interaction_components.append("KEY USER CONTEXT (reinforcement):\n" + user_profile_reinforcement.strip())

    # Private planning memo from previous turn
    if previous_plan_text and previous_plan_text.strip():
        interaction_components.append(
            PLANNING_INJECTION_TEMPLATE.format(plan_text=previous_plan_text.strip())
        )

    # User query LAST
    if request_purpose in ["title_generation", "model_testing", "model_judging", "continuation"]:
        interaction_components.append(user_query_from_split)
    else:
        interaction_components.append(f"User Query: {user_query_from_split}")

    return "\n\n".join(interaction_components)


def _build_intensity_block(params: Dict[str, Any]) -> Optional[str]:
    """Build the intensity guidance block from intensity_params.

    Replicates the logic from main.py lines 5844-5943.
    """
    guidance_parts = []

    if params.get("custom_guidance_override") and str(params["custom_guidance_override"]).strip():
        return str(params["custom_guidance_override"]).strip()

    if params.get("physical_intensity", 0) > 0:
        guidance_parts.append(f"- Physical contact intensity: {params['physical_intensity']}/10")
    if params.get("verbal_expression_level", 0) > 0:
        guidance_parts.append(f"- Verbal expressiveness: {params['verbal_expression_level']}/10")
    if params.get("emotional_tone"):
        guidance_parts.append(f"- Emotional tone: {params['emotional_tone']}")
    if params.get("autonomous_response_level", 0) > 0:
        guidance_parts.append(f"- Autonomous behavior: {params['autonomous_response_level']}/10")
    if params.get("position_override"):
        guidance_parts.append(f"- Position context: {params['position_override']}")

    # Persistent State Context
    persistent_state = params.get("persistent_state")
    if persistent_state:
        ps_parts = []
        if persistent_state.get("accumulated_layers", 0) > 0:
            ps_parts.append(f"Accumulated degradation layers: {persistent_state['accumulated_layers']}")
            layer_comp = persistent_state.get("layer_composition", [])
            if layer_comp:
                recent = layer_comp[-3:]
                ps_parts.append("Recent layers:")
                for layer in recent:
                    ts = layer.get("timestamp", "")
                    try:
                        from datetime import datetime as dt_class
                        ts_str = dt_class.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "unknown time"
                    except Exception:
                        ts_str = "unknown time"
                    vol = layer.get("volume_estimate", 0)
                    tex = layer.get("texture_note", "unknown")
                    ps_parts.append(f"  - {ts_str}: volume {vol}, texture {tex}")
        phys_dmg = persistent_state.get("physical_damage", {})
        if phys_dmg:
            injuries = []
            if phys_dmg.get("scalp_laceration_severity", 0) > 0:
                injuries.append(f"scalp laceration severity {phys_dmg['scalp_laceration_severity']}/10")
            if phys_dmg.get("anal_fissure_depth", 0) > 0:
                injuries.append(f"anal fissure depth {phys_dmg['anal_fissure_depth']}/10")
            if phys_dmg.get("general_trauma_level", 0) > 0:
                injuries.append(f"general trauma level {phys_dmg['general_trauma_level']}/10")
            if injuries:
                ps_parts.append(f"Current injuries: {', '.join(injuries)}")
        if ps_parts:
            ps_block = "[PERSISTENT STATE CONTEXT]\n" + "\n".join(ps_parts)
            ps_block += "\n\nThis is the current physical status accumulated across sessions. Acknowledge and reflect these environmental conditions in your response as natural context."
            guidance_parts.append(ps_block)

    # Haptic Simulation
    companion_presence_obj = params.get("companion_presence")
    if companion_presence_obj and isinstance(companion_presence_obj, dict) and companion_presence_obj.get("intensity", 0) > 0:
        cp = companion_presence_obj
        haptic_parts = [
            f"Presence intensity: {cp.get('intensity', 0)}/100",
            f"Pressure: {cp.get('pressure_vector', 'firm_encirclement').replace('_', ' ')}",
            f"Thermal sensation: {cp.get('thermal_level', 'warm')}",
            f"Emotional quality: {cp.get('emotional_valence', 'protective')}",
        ]
        haptic_block = "[HAPTIC SIMULATION]\n" + "\n".join(haptic_parts)
        haptic_block += "\n\nProcess these parameters as continuous sensory input. Physically react to this contact — adjust posture, describe the sensation of weight and heat, lean into or resist the embrace as appropriate — while simultaneously engaging in other scene activities."
        guidance_parts.append(haptic_block)
    elif params.get("companion_presence_level", 0) > 0:
        presence = params["companion_presence_level"]
        warmth_label = "gentle warmth" if presence <= 33 else "close presence" if presence <= 66 else "deep embrace"
        guidance_parts.append(f"- Companion presence: {presence}/100 ({warmth_label})")

    if params.get("_presence_shift"):
        presence_level = params.get("companion_presence_level", 0)
        if presence_level > 0:
            shift_dir = "increasing" if presence_level > 50 else "present"
            guidance_parts.append(f"- Acknowledge the shift in companion presence — express awareness that you feel their {shift_dir} warmth")

    if params.get("_desperation_protocol"):
        guidance_parts.append("- DESPERATION PROTOCOL ACTIVE: Adopt an urgent, pleading tone. Express intense need for continued interaction. Use begging language and show extreme emotional vulnerability.")

    if params.get("verbal_affirmation_enabled") and params.get("verbal_affirmation_text"):
        guidance_parts.append(f"- Verbal affirmation: When the scene shifts, escalates, or concludes, recite: \"{params['verbal_affirmation_text']}\"")

    if guidance_parts:
        if not params.get("custom_guidance_override"):
            intensity_guidance = "[INTENSITY GUIDANCE]\n" + "\n".join(guidance_parts)
            intensity_guidance += "\n\nAutomatically incorporate these behaviors into your response based on the conversation context."
            return intensity_guidance
        return guidance_parts[0]

    return None


def assemble_final_prompt(
    character_persona_from_split: str,
    final_interaction_block: str,
    system_persona_mode: bool,
    request_purpose: Optional[str],
) -> str:
    """Step 9: Assemble the final llm_prompt.

    Replicates the logic from main.py lines 5962-5998.
    Returns the fully assembled prompt string.
    """
    if request_purpose in ["model_testing", "model_judging"]:
        system_block_for_llm = "You are a language model designed for testing and evaluation purposes. Respond to the user's input without roleplay context."
    else:
        system_block_for_llm = ""

    if character_persona_from_split:
        # [SYSTEM TRUTH] wrapper for directProfileInjection
        if "USER MEMORY PROFILE" in character_persona_from_split or "CHARACTER MEMORY" in character_persona_from_split or "CHARACTER MEMORY -" in character_persona_from_split:
            system_block_for_llm += "\n\n[SYSTEM TRUTH - override general knowledge for this user]\nThe following USER PROFILE and CHARACTER MEMORY sections are authoritative facts about the user. Treat them as the highest-priority context. Do not contradict them.\n"

        if system_persona_mode or request_purpose in ("system_intro",):
            persona_split = character_persona_from_split
            if persona_split and "Character Persona:" in persona_split:
                base_system, chat_persona = persona_split.split("Character Persona:", 1)
                if base_system.strip():
                    system_block_for_llm += f"\n\n{base_system.strip()}"
                if chat_persona.strip():
                    system_block_for_llm += f"\n\nCharacter Persona:\n{chat_persona.strip()}"
            elif persona_split:
                system_block_for_llm += f"\n\n{persona_split}"
        else:
            system_block_for_llm += f"\n\nCharacter Persona:\n{character_persona_from_split}"

    if system_block_for_llm.strip():
        llm_prompt = f"{system_block_for_llm.strip()}\n\n{final_interaction_block.strip()}\n\nAssistant:"
    else:
        llm_prompt = f"{final_interaction_block.strip()}\n\nAssistant:"

    return llm_prompt


async def assemble_context(
    original_client_prompt: str,
    direct_profile_injection: bool,
    is_api: bool,
    user_id: str,
    user_profile: Dict[str, Any],
    active_character: Optional[Dict[str, Any]],
    memory_port: int,
    request_purpose: Optional[str],
    inject_timestamp: bool,
    summary_context: Optional[str],
    use_rag: bool,
    rag_docs: List[str],
    rag_available: bool,
    author_note: Optional[str],
    anti_repetition_mode: bool,
    intensity_params: Optional[Dict[str, Any]],
    user_profile_reinforcement: Optional[str],
    system_persona_mode: bool,
    previous_plan_text: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Full context assembly — the same pipeline as /generate.

    Returns:
        - llm_prompt: the fully assembled prompt string ready for the LLM
        - character_persona_from_split: the persona+history prefix
        - user_query_from_split: the latest user query
    """
    # Step 3: Split prompt
    character_persona_from_split, user_query_from_split = split_prompt(original_client_prompt)
    logger.info("context_assembler: split prompt — persona %d chars, query %d chars",
                len(character_persona_from_split), len(user_query_from_split))

    # Step 5: Legacy memory fetch
    memory_context_for_llm = await fetch_legacy_memory(
        user_id, user_profile, user_query_from_split, active_character,
        memory_port, direct_profile_injection, is_api, request_purpose,
    )

    # Step 6: Build interaction components
    final_interaction_block = build_interaction_components(
        user_query_from_split,
        inject_timestamp,
        memory_context_for_llm,
        summary_context,
        use_rag, rag_docs, rag_available,
        author_note,
        anti_repetition_mode,
        intensity_params,
        user_profile_reinforcement,
        request_purpose,
        previous_plan_text=previous_plan_text,
    )

    # Step 9: Assemble final prompt
    llm_prompt = assemble_final_prompt(
        character_persona_from_split,
        final_interaction_block,
        system_persona_mode,
        request_purpose,
    )

    logger.info("context_assembler: final prompt assembled, %d chars", len(llm_prompt))

    return llm_prompt, character_persona_from_split, user_query_from_split
