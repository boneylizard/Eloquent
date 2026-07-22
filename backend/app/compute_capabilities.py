import os
import sys


TRUE_VALUES = {"1", "true", "yes", "on"}


def force_cpu_mode() -> bool:
    """Allow diagnostics and users to disable CUDA without disabling Mirid."""
    return os.environ.get("MIRID_FORCE_CPU", "").strip().lower() in TRUE_VALUES


def disable_incompatible_torchao() -> None:
    """Keep Transformers from importing the optional, incompatible TorchAO build."""
    sys.modules.setdefault("torchao", None)
