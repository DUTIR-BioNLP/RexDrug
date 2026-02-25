#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DDI13 dataset to SFT instruction format.

Input format (per line, JSON object):
{
  "id": 0,
  "context": "sentence text...",
  "entities": [{"entity": "drug1", "category": "drug"}, ...],
  "relations": [{"type": "effect", "ent_sets": ["drug1", "drug2"]}, ...]
}

Output format (JSONL for ms-swift):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]"}
  ]
}

Usage:
    python process.py --input ../DDI2013/train.jsonl --output train.jsonl
    python process.py --input ../DDI2013/test.jsonl --output test.jsonl
"""

import json
import argparse
import os
from typing import Any, Dict, List

# System prompt
SYSTEM_PROMPT = (
    "You are an expert in biomedical relation extraction, specializing in identifying "
    "and classifying drug-drug interaction (DDI) relationships."
)

# User prompt template
USER_PROMPT_TEMPLATE = (
    "Identify all possible drug-drug interaction relationships mentioned in the target sentence. "
    "Use only the information in the sentence. Classify each interaction into exactly one of the following types:\n"
    "'effect': Pharmacodynamic or clinical effect.\n"
    "'advise': Clinical advice/warning/contraindication.\n"
    "'int': Interaction is stated but unspecified/unclear category.\n"
    "'mechanism': Pharmacokinetic or mechanistic basis.\n"
    "'NO_COMB': No drug-drug interaction is mentioned in the sentence.\n\n"
    "Target sentence: {sentence}\n\n"
    "Return the result strictly as JSON using the schema: [{'type': <interaction_type>, 'ent_set': [<ent1>, <ent2>]}]. "
    "Do not include any extra text or explanation."
)

# Valid relation types (lowercase)
VALID_TYPES = {"effect", "advise", "int", "mechanism"}


def to_single_quote_json(obj: Any) -> str:
    """Convert Python object to single-quote JSON string."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace('"', "'")


def process_relations(relations: List[Dict]) -> List[Dict[str, Any]]:
    """
    Process relations from input format to output format.
    Filter out 'other' type (maps to NO_COMB).
    """
    results = []

    for rel in relations:
        rel_type = rel.get("type", "").lower().strip()
        ent_sets = rel.get("ent_sets", [])

        # Skip 'other' type - these are non-interactions
        if rel_type == "other" or rel_type not in VALID_TYPES:
            continue

        # Normalize entity set
        ent_set = []
        seen = set()
        for ent in ent_sets:
            if isinstance(ent, str) and ent.strip():
                ent_clean = ent.strip()
                if ent_clean not in seen:
                    seen.add(ent_clean)
                    ent_set.append(ent_clean)

        if len(ent_set) >= 2:
            results.append({
                "type": rel_type,
                "ent_set": ent_set
            })

    return results


def convert_to_messages(data: Dict) -> Dict:
    """Convert a single data item to messages format."""
    sentence = data.get("context", "")
    relations = data.get("relations", [])

    # Process relations
    output_rels = process_relations(relations)

    # If no valid relations, output NO_COMB
    if not output_rels:
        output_rels = [{"type": "NO_COMB", "ent_set": []}]

    # Format output
    output_str = to_single_quote_json(output_rels)

    # Create user content (use replace instead of format to avoid brace conflicts)
    user_content = USER_PROMPT_TEMPLATE.replace("{sentence}", sentence)

    # Create messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_str}
    ]

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="Convert DDI13 dataset to SFT format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL file path")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSONL file path")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = convert_to_messages(data)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} examples -> {args.output}")


if __name__ == "__main__":
    main()

"""
Example usage (run from RexDrug root directory):

# Convert training data
python datasets/DDI13/sft_only_re/process.py \
    --input datasets/DDI13/DDI2013/train.jsonl \
    --output datasets/DDI13/sft_only_re/train.jsonl

# Convert test data
python datasets/DDI13/sft_only_re/process.py \
    --input datasets/DDI13/DDI2013/test.jsonl \
    --output datasets/DDI13/sft_only_re/test.jsonl
"""
