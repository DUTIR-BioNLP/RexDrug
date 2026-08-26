"""Compatibility exports for the shared DDI13 prompt definitions."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt.py"
_SPEC = spec_from_file_location("rexdrug_ddi13_prompts", _PROMPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load DDI13 prompts from {_PROMPT_PATH}")

_PROMPTS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROMPTS)

for _name in _PROMPTS.__all__:
    globals()[_name] = getattr(_PROMPTS, _name)

# Preserve the historical names for downstream scripts that still import them.
COT_TEACHER_PROMPT = RE_COT_TEMPLATE
EVALUATION_PROMPT = evaluation_prompt

__all__ = [*_PROMPTS.__all__, "COT_TEACHER_PROMPT", "EVALUATION_PROMPT"]
