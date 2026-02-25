#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a JSONL dataset for drug-combination extraction into a single JSON file
with records in the form:
{
  "instruction": ...,
  "input": ...,
  "output": "[{'type': 'COMB', 'ent_set': ['a','b']}]"
}
"""
import json
import argparse
from typing import Any, Dict, List

INSTRUCTION = "You are an expert in biomedical relation extraction, specializing in identifying and classifying drug combination relationships."

# NOTE: Escaped literal braces with double {{ }}
INPUT_TEMPLATE = (
    "Identify all possible drug combination relationships mentioned in the target sentence, and determine their combined usage effect category. "
    "You may refer to the surrounding paragraph for contextual reasoning. Possible relationship types include:\n"
    "'POS': The drug combination has a positive or synergistic therapeutic effect.\n"
    "'NEG': The drug combination has a negative or adverse effect.\n"
    "'COMB': The drug combination is used together without clearly positive or negative effects.\n"
    "'NO_COMB': No drug combination relationship is mentioned in the target sentence.\n\n"
    "Target sentence: {sentence}\n"
    "Context paragraph: {paragraph}\n\n"
    "Return the result strictly in JSON format, using the following schema: [{{'type': <relationship_type>, 'ent_set': [<drug1>, <drug2>, ...]}}]. "
    "Do not include any extra text or explanation."
)

ALLOWED_TYPES = {"POS", "NEG", "COMB", "NO_COMB"}

def stringify_single_quotes(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace('"', "'")

def build_span_map(spans: List[Dict[str, Any]]) -> Dict[Any, str]:
    return {sp.get("span_id"): sp.get("text", "") for sp in (spans or [])}

def extract_rels(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rels = obj.get("rels") or []
    if not isinstance(rels, list):
        return out

    span_map = build_span_map(obj.get("spans") or [])

    for rel in rels:
        rtype = (rel.get("class") or rel.get("type") or rel.get("label") or "").upper()
        if rtype not in ALLOWED_TYPES:
            continue
        ents: List[str] = []
        if isinstance(rel.get("entities"), list):
            for e in rel["entities"]:
                if isinstance(e, (str, int, float)):
                    ents.append(str(e))
        elif isinstance(rel.get("spans"), list) and span_map:
            for sid in rel["spans"]:
                if sid in span_map:
                    ents.append(span_map[sid])
        # dedupe preserve order
        seen = set()
        ent_set: List[str] = []
        for e in ents:
            if e not in seen:
                seen.add(e)
                ent_set.append(e)
        out.append({"type": rtype, "ent_set": ent_set})
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert JSONL to instruction-style JSON.")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--single-quotes", action="store_true", help="Render output field using single quotes like the example.")
    ap.add_argument("--emit-no-comb", dest="emit_no_comb", action="store_true", default=True)
    ap.add_argument("--no-emit-no-comb", dest="emit_no_comb", action="store_false")
    args = ap.parse_args()

    items = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            sentence = obj.get("sentence", "")
            paragraph = obj.get("paragraph", "")

            rel_items = extract_rels(obj)
            if not rel_items:
                if args.emit_no_comb:
                    rel_items = [{"type": "NO_COMB", "ent_set": []}]
                else:
                    rel_items = []

            output_str = stringify_single_quotes(rel_items) if args.single_quotes else json.dumps(rel_items, ensure_ascii=False)

            # Build messages format
            messages = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": INPUT_TEMPLATE.format(sentence=sentence, paragraph=paragraph)},
                {"role": "assistant", "content": output_str}
            ]
            items.append({"messages": messages})

    # Output as JSONL
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(items)} examples -> {args.output}")

if __name__ == "__main__":
    main()

