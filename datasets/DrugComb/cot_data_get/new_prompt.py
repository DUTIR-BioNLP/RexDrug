"""Compatibility exports for the shared DrugComb prompt definitions."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt.py"
_SPEC = spec_from_file_location("rexdrug_drugcomb_prompts", _PROMPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load DrugComb prompts from {_PROMPT_PATH}")

_PROMPTS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROMPTS)

for _name in _PROMPTS.__all__:
    globals()[_name] = getattr(_PROMPTS, _name)

cot_teacher_prompt = RE_COT_TEMPLATE

__all__ = [*_PROMPTS.__all__, "cot_teacher_prompt"]
