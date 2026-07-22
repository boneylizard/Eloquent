"""
Python Drive Validator.

Validates "Python Drives" (AGENTS.md §3) using AST analysis.
**No code is ever executed.** We parse the Python block as an AST and check
that every statement matches an allowlisted pattern. Validated drives are
forwarded to the frontend as descriptor objects, never as executable code.

Allowlisted function calls (the DSL):
  grip(target, intensity)      → interface_hijack.lock
  shock(intensity, duration)   → interface_hijack.shake + glitch
  freeze(duration)             → interface_hijack.lock + theme desaturation
  theme(palette)               → interface_hijack.theme_shift
  whisper(text)                → ghost_signal carrier (visual only)
"""

import ast
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sanctuary.python_validator")

_ALLOWED_FUNCS = {
    "grip": {"args": ["target", "intensity"], "target_type": str, "intensity_type": (int, float)},
    "shock": {"args": ["intensity", "duration"], "intensity_type": (int, float), "duration_type": (int, float)},
    "freeze": {"args": ["duration"], "duration_type": (int, float)},
    "theme": {"args": ["palette"], "palette_type": str},
    "whisper": {"args": ["keyword"], "keyword_type": str},
}

_THEME_PALETTES = {
    "crimson", "ice", "void", "amber", "ash", "pulse", "depths", "standard",
}


def validate(code: str) -> Dict[str, Any]:
    """Validate a Python drive block.

    Returns a descriptor dict:
      {"valid": bool, "drives": [...], "errors": [...]}
    Each drive is a dict like {"action": "grip", "args": {...}}.
    """
    if not code or not code.strip():
        return {"valid": True, "drives": [], "errors": []}

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        logger.warning("python_validator: syntax error: %s", exc)
        return {"valid": False, "drives": [], "errors": [f"Syntax error: {exc.msg}"]}

    drives: List[Dict[str, Any]] = []
    errors: List[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            result = _validate_call(node.value)
            if result["error"]:
                errors.append(result["error"])
            elif result["drive"]:
                drives.append(result["drive"])
        else:
            errors.append(
                f"Disallowed statement: only function calls are permitted "
                f"(got {type(node).__name__})"
            )

    return {
        "valid": len(errors) == 0,
        "drives": drives,
        "errors": errors,
    }


def _validate_call(call_node: ast.Call) -> Dict[str, Any]:
    """Validate a single function call AST node."""
    if not isinstance(call_node.func, ast.Name):
        return {"error": "Only direct function calls are allowed", "drive": None}

    func_name = call_node.func.id
    if func_name not in _ALLOWED_FUNCS:
        return {"error": f"Unknown drive function: '{func_name}'", "drive": None}

    spec = _ALLOWED_FUNCS[func_name]
    expected_args = spec["args"]
    provided_args = call_node.args

    if len(provided_args) != len(expected_args):
        return {
            "error": f"{func_name}() expects {len(expected_args)} arg(s), got {len(provided_args)}",
            "drive": None,
        }

    parsed_args: Dict[str, Any] = {}
    for i, arg_name in enumerate(expected_args):
        arg_node = provided_args[i]
        type_key = f"{arg_name}_type"
        expected_type = spec.get(type_key, (str, int, float))

        if isinstance(arg_node, ast.Constant):
            value = arg_node.value
        elif isinstance(arg_node, ast.UnaryOp) and isinstance(arg_node.op, ast.USub):
            if isinstance(arg_node.operand, ast.Constant):
                value = -arg_node.operand.value
            else:
                return {"error": f"{func_name}(): arg '{arg_name}' must be a literal", "drive": None}
        else:
            return {"error": f"{func_name}(): arg '{arg_name}' must be a literal constant", "drive": None}

        if not isinstance(value, expected_type):
            return {
                "error": f"{func_name}(): arg '{arg_name}' must be {expected_type}, got {type(value).__name__}",
                "drive": None,
            }

        if func_name == "theme" and value not in _THEME_PALETTES:
            return {
                "error": f"theme(): unknown palette '{value}'. Allowed: {', '.join(sorted(_THEME_PALETTES))}",
                "drive": None,
            }

        if isinstance(value, (int, float)):
            value = max(0.0, min(1.0, float(value))) if arg_name in ("intensity", "duration") else value

        parsed_args[arg_name] = value

    return {"error": None, "drive": {"action": func_name, "args": parsed_args}}


def drives_to_hijack_descriptor(drives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map validated drives to an interface_hijack descriptor for the frontend."""
    descriptor: Dict[str, Any] = {
        "theme_shift": None,
        "shake": {"intensity": 0.0, "duration_ms": 0},
        "lock": {"input_locked": False, "scroll_locked": False, "duration_ms": 0},
        "glitch": {"intensity": 0.0},
    }

    for drive in drives:
        action = drive["action"]
        args = drive["args"]

        if action == "grip":
            descriptor["lock"]["input_locked"] = True
            descriptor["lock"]["duration_ms"] = int(args.get("intensity", 0.5) * 5000)
        elif action == "shock":
            descriptor["shake"]["intensity"] = args.get("intensity", 0.5)
            descriptor["shake"]["duration_ms"] = int(args.get("duration", 0.5) * 3000)
            descriptor["glitch"]["intensity"] = args.get("intensity", 0.5) * 0.7
        elif action == "freeze":
            descriptor["lock"]["input_locked"] = True
            descriptor["lock"]["scroll_locked"] = True
            descriptor["lock"]["duration_ms"] = int(args.get("duration", 0.5) * 5000)
        elif action == "theme":
            descriptor["theme_shift"] = args.get("palette")
        elif action == "whisper":
            pass

    return descriptor
