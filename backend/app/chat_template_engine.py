# backend/app/chat_template_engine.py
"""
Custom Jinja chat-template engine for local GGUF models.

Stores per-model Jinja templates (loaded from ~/.LiangLocal/settings.json and
a small set of built-in defaults) and renders structured message arrays into
prompt strings.  This lets Eloquent use the exact chat template that LM Studio
uses for a model, which fixes formatting drift and runaway generation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, BaseLoader, TemplateError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in defaults (pre-seeded templates).  Users can override any of these
# by creating an entry with a matching pattern in settings.json under the key
# ``modelChatTemplates``.
# ---------------------------------------------------------------------------

_QWEN3_FROGGERIC_V21_TEMPLATE = r"""{%- set template_version = "qwen3.6-froggeric-v21.3" %}
{%- set _tool_format = tool_call_format if tool_call_format is defined else 'xml' %}
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- set add_vision_id = add_vision_id if add_vision_id is defined else false %}
{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}
{%- set auto_disable_thinking_with_tools = auto_disable_thinking_with_tools if auto_disable_thinking_with_tools is defined else false %}
{%- set _preserve_thinking = preserve_thinking if preserve_thinking is defined else true %}
{%- set max_tool_arg_chars = max_tool_arg_chars if max_tool_arg_chars is defined else 0 %}
{%- set max_tool_response_chars = max_tool_response_chars if max_tool_response_chars is defined else 0 %}
{%- set _has_tools = (tools is defined and tools and tools is iterable and tools is not mapping) %}
{%- set ns_state = namespace(thinking=enable_thinking) %}
{%- if auto_disable_thinking_with_tools and _has_tools %}
    {%- set ns_state.thinking = false %}
{%- endif %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if item is mapping %}
                {%- if item.type == 'image' or 'image' in item or 'image_url' in item %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain images.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set image_count.value = image_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Picture ' ~ image_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                {%- elif item.type == 'video' or 'video' in item %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain videos.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set video_count.value = video_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Video ' ~ video_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                {%- elif 'text' in item %}
                    {{- item.text }}
                {%- else %}
                    {{- raise_exception('Unexpected item type in content.') }}
                {%- endif %}
            {%- else %}
                {{- item | string }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set _first_role = messages[0].role %}
{%- if _first_role == 'system' or _first_role == 'developer' %}
    {%- set _sys_msg = messages[0] %}
    {%- set _msgs = messages[1:] %}
{%- else %}
    {%- set _sys_msg = none %}
    {%- set _msgs = messages %}
{%- endif %}
{%- set _sc = '' %}
{%- if _sys_msg is not none %}
    {%- set _sc = render_content(_sys_msg.content, false, true) | trim %}
    {%- if '<|think_off|>' in _sc %}
        {%- set ns_state.thinking = false %}
        {%- set _sc = _sc.split('<|think_off|>') | join('') | trim %}
    {%- elif '<|think_on|>' in _sc %}
        {%- set ns_state.thinking = true %}
        {%- set _sc = _sc.split('<|think_on|>') | join('') | trim %}
    {%- endif %}
{%- endif %}
{%- if _has_tools %}
    {{- '<|im_start|>system\n' }}
    {{- '# Tools\n\nYou have access to the following functions:\n\n<tools>' }}
    {%- for tool in tools %}
        {{- '\n' }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- '\n</tools>' }}
    {%- set tool_instructions %}
If you choose to call a function ONLY reply in the following format with NO suffix:

{%- if _tool_format == 'json' %}
<think>
Brief explanation of tool call
</think>
<tool_call>
{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1", "example_parameter_2": "This is the value for the second parameter"}}
</tool_call>
{%- else %}
<think>
Brief explanation of tool call
</think>
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>
{%- endif %}

<IMPORTANT>
Reminder:
- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.
- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.
{%- if _tool_format == 'json' %}
- Function calls MUST follow the specified format: a single JSON object with "name" and "arguments" keys inside <tool_call></tool_call> XML tags.
{%- else %}
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.
{%- endif %}
- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.
{%- if _tool_format == 'json' %}
- The <tool_call> tag MUST be at the very beginning of a new line, with NO spaces or indentation before it.
{%- else %}
- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.
{%- endif %}
- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.
- If you have all necessary data, provide your final answer directly to the user without any tool call.
</IMPORTANT>
    {%- endset %}
    {{- '\n\n' ~ tool_instructions | trim }}
    {%- if _sc %}
        {{- '\n\n' + _sc }}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if _sc %}
        {{- '<|im_start|>system\n' + _sc + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set _last_idx = _msgs | length - 1 %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=_last_idx) %}
{%- for message in _msgs[::-1] %}
    {%- set index = (_msgs | length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == 'user' %}
        {%- set _rc = render_content(message.content, false) | trim %}
        {%- if not (_rc.startswith('<tool_response>') and _rc.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {%- if _last_idx > 50 %}
        {%- set ns.last_query_index = _last_idx %}
    {%- else %}
        {%- set ns.last_query_index = 0 %}
    {%- endif %}
{%- endif %}
{%- set ns2 = namespace(prev_role='', consecutive_failures=0) %}
{%- for message in _msgs %}
    {%- set is_system = (message.role == "system" or message.role == "developer") %}
    {%- set content = render_content(message.content, true, is_system) | trim %}
    {%- if is_system or message.role == 'user' %}
        {%- if '<|think_off|>' in content %}
            {%- set ns_state.thinking = false %}
            {%- set content = content.split('<|think_off|>') | join('') | trim %}
        {%- elif '<|think_on|>' in content %}
            {%- set ns_state.thinking = true %}
            {%- set content = content.split('<|think_on|>') | join('') | trim %}
        {%- endif %}
    {%- endif %}
    {%- if is_system %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == 'user' %}
        {%- set ns2.consecutive_failures = 0 %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- if message.reasoning_content is string %}
                {%- set reasoning_content = message.reasoning_content %}
            {%- else %}
                {%- set reasoning_content = message.reasoning_content | string %}
            {%- endif %}
        {%- elif message.thinking is defined and message.thinking is not none %}
            {%- if message.thinking is string %}
                {%- set reasoning_content = message.thinking %}
            {%- else %}
                {%- set reasoning_content = message.thinking | string %}
            {%- endif %}
        {%- else %}
            {%- set _think_end = '' %}
            {%- if content.startswith('</think>') %}
                {%- set _think_end = '</think>' %}
            {%- elif content.startswith('</thinking>') %}
                {%- set _think_end = '</thinking>' %}
            {%- elif '\n</think>' in content %}
                {%- set _think_end = '\n</think>' %}
            {%- elif '\n</thinking>' in content %}
                {%- set _think_end = '\n</thinking>' %}
            {%- elif '\n</ think>' in content %}
                {%- set _think_end = '\n</ think>' %}
            {%- elif '\n</think >' in content %}
                {%- set _think_end = '\n</think >' %}
            {%- endif %}
            {%- if _think_end %}
                {%- if 'thinking' in _think_end %}
                    {%- set _think_start = '<thinking>' %}
                {%- else %}
                    {%- set _think_start = '<think>' %}
                {%- endif %}
                {%- set reasoning_content = content.split(_think_end)[0].rstrip('\n') %}
                {%- if _think_start in reasoning_content %}
                    {%- set reasoning_content = reasoning_content.split(_think_start)[-1].lstrip('\n') %}
                {%- endif %}
                {%- set content = content.split(_think_end)[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content | trim %}
        {%- if (_preserve_thinking or loop.index0 > ns.last_query_index) and reasoning_content %}
            {{- '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>assistant\n' + content }}
        {%- endif %}
        {%- if message.tool_calls is defined and message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined and tool_call.function is not none %}
                    {%- set tc = tool_call.function %}
                {%- else %}
                    {%- set tc = tool_call %}
                {%- endif %}
                {%- if _tool_format == 'json' %}
                    {%- if not loop.first or content | trim %}
                        {{- '\n\n' }}
                    {%- endif %}
                    {%- set _args = '{}' %}
                    {%- if tc.arguments is defined and tc.arguments is not none %}
                        {%- if tc.arguments is mapping %}
                            {%- set _args = tc.arguments | tojson %}
                        {%- elif tc.arguments is string and tc.arguments %}
                            {%- set _args = tc.arguments %}
                        {%- endif %}
                    {%- endif %}
                    {{- '<tool_call>\n{"name": ' }}{{- tc.name | tojson }}{{- ', "arguments": ' }}{{- _args }}{{- '}\n</tool_call>' }}
                {%- else %}
                    {%- if loop.first %}
                        {%- if content | trim %}
                            {{- '\n\n<tool_call>\n<function=' + tc.name + '>\n' }}
                        {%- else %}
                            {{- '<tool_call>\n<function=' + tc.name + '>\n' }}
                        {%- endif %}
                    {%- else %}
                        {{- '\n\n<tool_call>\n<function=' + tc.name + '>\n' }}
                    {%- endif %}
                    {%- if tc.arguments is defined and tc.arguments is not none %}
                        {%- if tc.arguments is mapping %}
                            {%- for args_name, args_value in tc.arguments.items() %}
                                {{- '<parameter=' + args_name + '>\n' }}
                                {%- if args_value is mapping or (args_value is sequence and args_value is not string) %}
                                    {%- set _av = args_value | tojson %}
                                {%- else %}
                                    {%- set _av = args_value | string %}
                                {%- endif %}
                                {%- if max_tool_arg_chars > 0 and _av | length > max_tool_arg_chars %}
                                    {{- _av[:max_tool_arg_chars] + '\n[TRUNCATED — original length ' ~ (_av | length | string) ~ ' chars]' }}
                                {%- else %}
                                    {{- _av }}
                                {%- endif %}
                                {{- '\n</parameter>\n' }}
                            {%- endfor %}
                        {%- elif tc.arguments is string and tc.arguments %}
                            {{- tc.arguments }}
                        {%- endif %}
                    {%- endif %}
                    {{- '</function>\n</tool_call>' }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == 'tool' %}
        {%- set _content_lower = content | lower %}
        {%- set _content_head = _content_lower[:80] %}
        {%- if content | length < 500 and '$ ' not in content and 'took ' not in _content_lower and ('"error":' in _content_head or 'error:' in _content_head or 'err!' in _content_head or 'fatal:' in _content_head or 'exception:' in _content_head or 'traceback' in _content_head or 'command not found' in _content_head or 'invalid syntax' in _content_head or 'failed to' in _content_head) %}
            {%- set ns2.consecutive_failures = ns2.consecutive_failures + 1 %}
        {%- else %}
            {%- set ns2.consecutive_failures = 0 %}
        {%- endif %}
        {%- if ns2.prev_role != 'tool' %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {%- if max_tool_response_chars > 0 and content | length > max_tool_response_chars %}
            {%- set content = content[:max_tool_response_chars] + '\n[TRUNCATED — original length ' ~ (content | length | string) ~ ' chars]' %}
        {%- endif %}
        {{- '\n<tool_response>\n' + content }}
        {%- if ns2.consecutive_failures >= 2 %}
            {{- '\n\n⚠️ SYSTEM WARNING: ' ~ ns2.consecutive_failures ~ ' consecutive tool errors detected. Your previous approach is incorrect. You MUST use a fundamentally different approach or corrected arguments.' }}
        {%- elif ns2.consecutive_failures == 1 %}
            {{- '\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and retry with completely corrected arguments.' }}
        {%- endif %}
        {{- '\n</tool_response>' }}
        {%- if loop.last %}
            {{- '<|im_end|>\n' }}
        {%- else %}
            {%- set _next_role = _msgs[loop.index0 + 1].role %}
            {%- if _next_role != 'tool' %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>user\n[' + message.role + ']: ' + content + '<|im_end|>\n' }}
    {%- endif %}
    {%- set ns2.prev_role = message.role %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if not ns_state.thinking %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- elif ns2.consecutive_failures >= 2 %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"""

_GENERIC_CHAT_TEMPLATE = """{%- for message in messages %}
{%- if message.role in ['system', 'developer'] %}
{{- 'System: ' + message.content + '\n\n' }}
{%- elif message.role == 'user' %}
{{- 'User: ' + message.content + '\n' }}
{%- elif message.role == 'assistant' %}
{{- 'Assistant: ' + message.content + '\n\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- 'Assistant: ' }}{%- endif %}"""

_CHATML_CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + (message.role if message.role != 'developer' else 'system') + '\n' }}
{{- message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""

SELECTABLE_CHAT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "generic": {
        "patterns": ["__manual_generic__"],
        "template": _GENERIC_CHAT_TEMPLATE,
        "stop_tokens": ["\nUser:", "\nSystem:"],
    },
    "chatml": {
        "patterns": ["__manual_chatml__"],
        "template": _CHATML_CHAT_TEMPLATE,
        "stop_tokens": ["<|im_end|>", "<|im_start|>user"],
    },
}

DEFAULT_CHAT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "qwen3-froggeric-v21": {
        "patterns": [
            "qwen3.5",
            "qwen3.6",
            "qwen-3.5",
            "qwen-3.6",
            "huihui-qwen3.6",
            "claude-4.7-opus-abliterated",
        ],
        "template": _QWEN3_FROGGERIC_V21_TEMPLATE,
        "stop_tokens": ["<|im_end|>", "<|im_start|>user"],
    },
}

# Standard fallback stop tokens for custom templates that don't declare any.
DEFAULT_STOP_TOKENS = ["<|im_end|>", "<|im_start|>user", "</s>"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_path() -> Path:
    return Path.home() / ".LiangLocal" / "settings.json"


def _load_user_templates() -> Dict[str, Dict[str, Any]]:
    """Load modelChatTemplates from settings.json."""
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("modelChatTemplates") or {}
    except Exception as exc:
        logger.warning("Failed to load user chat templates from settings: %s", exc)
        return {}


def _compile_registry(templates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Validate and compile a template map into an ordered registry.
    Invalid entries are skipped with a warning.
    """
    registry: Dict[str, Dict[str, Any]] = {}

    for name, entry in templates.items():
        if not isinstance(entry, dict):
            logger.warning("Skipping invalid chat template entry %r", name)
            continue
        template_text = entry.get("template")
        if not template_text or not isinstance(template_text, str):
            logger.warning("Skipping chat template %r: missing 'template' string", name)
            continue
        patterns = entry.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [p.strip() for p in patterns.split(",") if p.strip()]
        if not isinstance(patterns, list):
            patterns = []
        # If no explicit patterns, allow matching by the registry key itself.
        if not patterns:
            patterns = [name.lower()]

        stop_tokens = entry.get("stop_tokens") or DEFAULT_STOP_TOKENS
        if isinstance(stop_tokens, str):
            stop_tokens = [s.strip() for s in stop_tokens.split(",") if s.strip()]
        if not isinstance(stop_tokens, list) or not stop_tokens:
            stop_tokens = DEFAULT_STOP_TOKENS
        try:
            env = Environment(loader=BaseLoader(), trim_blocks=False, lstrip_blocks=False)
            env.globals["raise_exception"] = _raise_exception
            compiled = env.from_string(template_text)
        except TemplateError as exc:
            logger.error("Failed to compile chat template %r: %s", name, exc)
            continue
        registry[name] = {
            "patterns": [p.lower() for p in patterns if isinstance(p, str)],
            "template": template_text,
            "compiled": compiled,
            "stop_tokens": stop_tokens,
        }
    return registry


def _raise_exception(message: str) -> None:
    """Jinja callable: raise a TemplateError with the supplied message."""
    raise TemplateError(message)


def _normalize_content(content: Any) -> Any:
    """Make sure message content is in a format the template macros expect."""
    if content is None:
        return ""
    if isinstance(content, (str, list, dict)):
        return content
    return str(content)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages so the Jinja template won't choke on odd shapes."""
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            continue
        out.append({
            "role": role,
            "content": _normalize_content(msg.get("content")),
            "reasoning_content": msg.get("reasoning_content") if isinstance(msg.get("reasoning_content"), str) else None,
            "tool_calls": msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else None,
        })
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(model_name: Optional[str], template_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find a chat template by an explicit selectable id or by model-name matching.
    User-defined templates take precedence over built-in defaults.
    Returns the compiled registry entry, or None if no match.
    """
    selected = (template_id or "").strip()
    user_registry = _compile_registry(_load_user_templates())

    if selected and selected != "model-default":
        if selected.startswith("custom:"):
            selected = selected.split(":", 1)[1]
        if selected in user_registry:
            return user_registry[selected]
        selectable_registry = _compile_registry(SELECTABLE_CHAT_TEMPLATES)
        return selectable_registry.get(selected)

    if not model_name:
        return None
    lower = model_name.lower()
    default_registry = _compile_registry(DEFAULT_CHAT_TEMPLATES)
    for registry in (user_registry, default_registry):
        for entry in registry.values():
            for pattern in entry["patterns"]:
                if pattern in lower:
                    return entry
    return None


def render(messages: List[Dict[str, Any]], model_name: Optional[str] = None, *,
           template_id: Optional[str] = None,
           add_generation_prompt: bool = True,
           enable_thinking: bool = False,
           add_vision_id: bool = False,
           preserve_thinking: bool = True,
           tools: Optional[List[Any]] = None) -> str:
    """
    Render *messages* using the custom template registered for *model_name*.

    Raises TemplateError if the template itself fails.  Callers should catch
    this and fall back to the legacy prompt path.
    """
    entry = lookup(model_name, template_id)
    if entry is None:
        raise TemplateError(f"No custom chat template registered for model {model_name!r}")

    normalized = _normalize_messages(messages)
    ctx = {
        "messages": normalized,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "add_vision_id": add_vision_id,
        "preserve_thinking": preserve_thinking,
        "tools": tools or [],
    }
    return entry["compiled"].render(ctx)


def render_with_stops(messages: List[Dict[str, Any]], model_name: Optional[str] = None,
                      *, template_id: Optional[str] = None,
                      **kwargs) -> Tuple[str, List[str]]:
    """
    Convenience: render and also return the configured stop tokens.
    If the template is not found, raises TemplateError.
    """
    entry = lookup(model_name, template_id)
    if entry is None:
        raise TemplateError(f"No custom chat template registered for model {model_name!r}")
    return render(messages, model_name, template_id=template_id, **kwargs), list(entry.get("stop_tokens") or DEFAULT_STOP_TOKENS)


def merge_backend_context(messages: List[Dict[str, Any]],
                          system_block: str,
                          interaction_block: str) -> List[Dict[str, Any]]:
    """
    Inject backend-assembled context (system truth, memory, RAG, etc.) into the
    structured message array before Jinja rendering.

    - *system_block* is prepended to the first system message (or a new system
      message is inserted at the front).
    - *interaction_block* is merged into the last user message, replacing the
      plain ``User Query:`` prefix if it is present.
    """
    if not messages:
        messages = [{"role": "user", "content": interaction_block or ""}]

    messages = _normalize_messages(messages)
    out = [dict(m) for m in messages]

    system_block = (system_block or "").strip()
    interaction_block = (interaction_block or "").strip()

    # Inject backend system context into the first system message.
    if system_block:
        sys_idx = next((i for i, m in enumerate(out) if m["role"] in ("system", "developer")), -1)
        if sys_idx >= 0:
            existing = str(out[sys_idx].get("content") or "").strip()
            if existing:
                out[sys_idx]["content"] = f"{system_block}\n\n{existing}"
            else:
                out[sys_idx]["content"] = system_block
        else:
            out.insert(0, {"role": "system", "content": system_block})

    # Inject interaction context into the last user message.
    if interaction_block:
        user_indices = [i for i, m in enumerate(out) if m["role"] == "user"]
        if user_indices:
            last_user_idx = user_indices[-1]
            existing = str(out[last_user_idx].get("content") or "").strip()
            # If the backend interaction block is just "User Query: X", prefer
            # the richer frontend user content; only keep the query marker if it
            # adds new information.
            if interaction_block.lower().startswith("user query:"):
                query_part = interaction_block.split(":", 1)[1].strip()
                if query_part and existing and query_part != existing:
                    out[last_user_idx]["content"] = f"{existing}\n\n[User Query]\n{query_part}"
                elif query_part and not existing:
                    out[last_user_idx]["content"] = query_part
            else:
                # The interaction block contains memory/RAG context.
                if existing:
                    out[last_user_idx]["content"] = f"{interaction_block}\n\n{existing}"
                else:
                    out[last_user_idx]["content"] = interaction_block

    return out
