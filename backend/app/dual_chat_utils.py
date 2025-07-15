"""
Dual Chat Utilities - Improved version with robust output processing
Handles formatting, cleaning and processing of LLM outputs in dual chat contexts
"""
import re
from typing import List, Optional


# Constants - using a less leak-prone end marker format
INTERNAL_END = "<|DONE|>"  # Changed to be shorter and less likely to leak
DUAL_CHAT_MARKER = r"\(?\|?—\s*The Assistant\s*\|?\)?"


def append_end_marker(user_content: str) -> str:
    """
    Assemble the final prompt:
      1) system prompt (sysMsg)
      2) user content (persona, memory, user query)
      3) end marker (<|DONE|>)
    """
    end = "<|DONE|>"
    # Trim whitespace around pieces to avoid accidental blank lines
    return f"{user_content.strip()}\n\n{end}"


def get_stop_sequences(orig: List[str]) -> List[str]:
    """Add the internal end marker to stop sequences."""
    if not orig:
        orig = []
    return orig + [INTERNAL_END]


def strip_end_marker(text: str) -> str:
    """Remove any occurrences of the internal end marker."""
    if not text:
        return ""
    return text.replace(INTERNAL_END, "")


def strip_thinking_markers(text: str) -> str:
    """
    Aggressively remove all control phrases and thinking markers.
    """
    if not text:
        return text
    
    # Remove thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>.*?(\n|$)', '', text)
    
    # Common phrase patterns that signal instruction leakage
    control_patterns = [
        # Remove lines with signal/end instructions
        r'^.*?\b(?:to signal|to indicate|to mark|to show|to signify|to let you know)\b.*?(?:completion|finished|complete|end|done).*?(?:\n|$)',
        
        # Meta-commentary about AI/assistant
        r'^.*?\b(?:so|this|here|this is so|this allows|this lets)\b.*?\b(?:the AI|the assistant|the model|you|the system)\b.*?(?:know|understand|see|recognize).*?(?:\n|$)',
        
        # System or assistant prefixes
        r'^(?:System|Assistant|AI):\s*(?:\n|$)',
        
        # Common auto-generated prefixes
        r'^(?:After-hours|Wind-Down|Coding Session|Conclusion)\b.*?(?:\n|$)',
        
        # NEW: Catch instances of the marker instruction leaking through
        r'^.*?(?:When you\'re done|end with|when finished|To finish).*?(?:\n|$)',
        
        # NEW: Catch specific cases like "the response must end with"
        r'^.*?(?:response must end|must end with|must be terminated|finish with).*?(?:\n|$)',
    ]
    
    # Apply each pattern with proper flags
    for pattern in control_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Process further only if we still have text
    if text.strip():
        # Handle full-line control instructions
        lines = text.split('\n')
        if lines and re.search(r'^to\s+(signal|indicate|finish|let|mark|show)', lines[0].lower().strip()):
            lines = lines[1:]  # Skip first line
        text = '\n'.join(lines)
        
        # Clean other control phrases with improved patterns
        text = re.sub(r'^[^\n]*?to\s+(?:signal|indicate|finish|return|mark|show|let).*?(?:response|answer|end|complete|stop|control|finished)[^.]*\.', '', text, flags=re.MULTILINE)
        text = re.sub(r'to\s+(?:signal|indicate|finish|return|mark|show|let).*?(?:response|answer|end|complete|stop|control|finished)[^.]*\.', '', text)
        
        # NEW: Extra cleanup for any remaining marker references
        text = re.sub(r'(?:When you\'re done|end with|when finished).*?(?:<\|DONE\|>|<\|END_OF_ANSWER\|>)[^.]*\.', '', text, flags=re.IGNORECASE)
    
    return text


def strip_ui_wrappers(text: str) -> str:
    """Remove UI-related elements from the text."""
    if not text:
        return ""
        
    lines = []
    for ln in text.splitlines():
        t = ln.strip()
        # Skip UI markers
        if re.match(DUAL_CHAT_MARKER, t):
            continue
        if t.startswith("```"):
            continue
        if re.match(r"^[\w\-.]+'s avatar$", t):
            continue
        lines.append(ln)
    
    return "\n".join(lines).strip()


def strip_generic_endings(text: str) -> str:
    """Remove common generic endings that AI models tend to add."""
    if not text:
        return ""
        
    generic_endings = [
        r"Feel free to share.*?I may have missed\.",
        r"Let me know if you have any.*?",
        r"I'd be happy to discuss.*?",
        r"What are your thoughts on this\?",
        r"Would you like me to elaborate.*?",
        # NEW: Additional patterns
        r"I hope this helps.*?",
        r"Please let me know if.*?",
        r"Is there anything else.*?",
    ]
    
    cleaned = text.strip()
    for pattern in generic_endings:
        cleaned = re.sub(pattern + r'$', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned


def needs_continuation(text: str, dual: bool=False) -> bool:
    """Determine if a response needs continuation."""
    if not text:
        return False
        
    # Has proper ending punctuation
    if re.search(r"[\.!?]\s*$", text):
        return False
        
    # Has dual-chat marker
    if dual and re.search(DUAL_CHAT_MARKER, text):
        return False
        
    return True


def build_continuation_prompt(model: str, clean: str, dual: bool=False) -> str:
    """Build a prompt for continuing an incomplete response."""
    if not clean:
        return ""
        
    if dual:
        tpl = (
            f"You are {model} in a conversation with another AI.\n"
            "Your last thought was cut off. Continue directly where you left off.\n"
            f"{clean}\n\nContinue:"
        )
    else:
        tpl = f"You are {model}. Continue your last answer:\n\n{clean}\n\nContinue."
    
    return append_end_marker(tpl)


def process_dual_chat_output(text: str, model: str) -> str:
    """
    Process and clean output from a dual chat interaction.
    
    Args:
        text: The raw text output from the model
        model: The name of the model that generated the text
        
    Returns:
        Cleaned and formatted text
    """
    if not text:
        return ""
    
    # Apply all cleaning functions in sequence
    out = strip_thinking_markers(text)
    out = strip_ui_wrappers(out)
    out = strip_end_marker(out)
    
    # Remove inline control phrases
    out = re.sub(r'\s*to (signal|indicate|let me know|finish|return).*?(response|answer|finished|control|complete|end|stop)[^.]*\.', ' ', out)
    
    # Normalize & split on the single marker
    out = re.sub(DUAL_CHAT_MARKER, "<<A>>", out)
    parts = [p.strip() for p in out.split("<<A>>") if p.strip()]
    
    # Apply generic ending removal to each part
    if len(parts) > 1:
        clean_parts = [strip_generic_endings(p) for p in parts]
        clean_parts = [p for p in clean_parts if p.strip()]  # Remove empty parts
        
        return "\n\n".join(f"{model}: {p}" if i>0 else p
                           for i,p in enumerate(clean_parts))
    
    result = parts[0] if parts else ""
    return strip_generic_endings(result)