"""Compatibility exports for the shared DrugComb training prompts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt.py"
_SPEC = spec_from_file_location("rexdrug_drugcomb_prompts", _PROMPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load DrugComb prompts from {_PROMPT_PATH}")

_PROMPTS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROMPTS)

SYSTEM_PROMPT = _PROMPTS.SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = _PROMPTS.USER_PROMPT_TEMPLATE
INPUT_TEMPLATE = USER_PROMPT_TEMPLATE

__all__ = ["SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "INPUT_TEMPLATE"]
