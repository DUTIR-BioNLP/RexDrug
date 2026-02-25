#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DDI13 CoT Data to SFT Instruction Format

This script converts the generated Chain-of-Thought data to the ms-swift
SFT (Supervised Fine-Tuning) instruction format with messages structure.

Input format (JSONL from get_cot_data.py):
{
    "idx": 1,
    "sentence": "...",
    "think": "reasoning content",
    "answer": "[{...}]",
    ...
}

Output format (JSONL for ms-swift SFT):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>...</think>\n\n<answer>...</answer>"}
    ]
}

Usage:
    python convert_cot_to_sft.py --input ./output/final_cot_xxx.jsonl --output ../sft_cot_data/train.jsonl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System prompt for DDI13 CoT reasoning
SYSTEM_PROMPT = (
    "You are an expert in biomedical relation extraction, specializing in sentence-level "
    "drug-drug interaction (DDI) extraction. You first think step by step in a clinically "
    "and pharmacologically interpretable way inside <think>...</think>, then output the "
    "final relation extraction result as JSON inside an <answer>...</answer> tag."
)

# User input template
INPUT_TEMPLATE = """Identify all drug-drug interaction (DDI) relations expressed in the target sentence, and assign the correct DDI relation type for each interacting entity pair. Possible relation types include:
'int': An interaction is stated but not further specified into mechanism/effect/advice.
'mechanism': The sentence describes how the interaction occurs (PK/PD mechanism).
'effect': The sentence describes the interaction outcome (e.g., exposure/toxicity/efficacy change) without explicit guidance.
'advice': The sentence provides explicit clinical guidance (e.g., avoid, contraindicated, dose adjust, monitor).
'NO_COMB': No interaction relation is expressed between any entity pair in the sentence.

Then provide your reasoning and final answer in two parts:
1) First, output your reasoning inside <think>...</think>.
   - Inside <think>, write four numbered sections:
     [1] Clinical / pharmacological scenario
     [2] Candidate entities and pair focus
     [3] Interaction reasoning and relation labeling
     [4] Extraction-oriented clinical summary
   - Under each section, use bullet points starting with "- ", with short, clinically/pharmacologically oriented sentences.
   - Keep the total reasoning concise (about 100-200 words).
2) Immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.
   - Inside <answer>, return a valid JSON array with the schema:
     [{{"type": "rel_type", "ent_set": ["ent1", "ent2"]}}]
   - Each ent_set must contain exactly two entity strings (a binary pair).
   - Do not include any extra text or explanation outside <answer>.

Target sentence: {sentence}"""


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def convert_to_sft_format(input_file: str, output_file: str) -> dict:
    """
    Convert DDI13 CoT data to SFT instruction format.

    Args:
        input_file: Path to input JSONL file (generated CoT data)
        output_file: Path to output JSONL file (SFT format)

    Returns:
        Statistics dictionary with conversion results
    """
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "failed_indices": []
    }

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            stats["total"] += 1

            try:
                data = json.loads(line.strip())

                # Get required fields
                sentence = data.get("sentence", "")
                think = data.get("think", "")
                answer = data.get("answer", "")

                # Validate required fields
                if not sentence:
                    stats["failed"] += 1
                    stats["failed_indices"].append(data.get("idx", line_num))
                    print(f"Warning: Line {line_num} missing sentence, skipping...")
                    continue

                if not think or not answer:
                    stats["failed"] += 1
                    stats["failed_indices"].append(data.get("idx", line_num))
                    print(f"Warning: Line {line_num} missing think or answer, skipping...")
                    continue

                # Build user content using template
                user_content = INPUT_TEMPLATE.replace("{sentence}", sentence)

                # Build assistant content with <think> and <answer> tags
                assistant_content = f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"

                # Create SFT format
                sft_data = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }

                # Write to output
                f_out.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
                stats["success"] += 1

            except json.JSONDecodeError as e:
                stats["failed"] += 1
                stats["failed_indices"].append(line_num)
                print(f"Error: Line {line_num} JSON decode error: {e}")

            except Exception as e:
                stats["failed"] += 1
                stats["failed_indices"].append(line_num)
                print(f"Error: Line {line_num} unexpected error: {e}")

    return stats


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert DDI13 CoT data to SFT instruction format"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input JSONL file path (CoT data from get_cot_data.py)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSONL file path (default: auto-generate with timestamp)"
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = script_dir / f"sft_cot_ddi13_{timestamp}.jsonl"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path

    print("=" * 60)
    print("DDI13 CoT to SFT Data Converter")
    print("=" * 60)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print("=" * 60)

    # Check input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Execute conversion
    stats = convert_to_sft_format(str(input_path), str(output_path))

    # Print statistics
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Total samples:   {stats['total']}")
    print(f"Success:         {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    if stats['failed'] > 0:
        failed_preview = stats['failed_indices'][:10]
        print(f"Failed indices:  {failed_preview}{'...' if len(stats['failed_indices']) > 10 else ''}")
    print(f"\nOutput saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Example usage (run from RexDrug root directory):

# Convert generated CoT data to SFT format for training
python datasets/DDI13/cot_data_get/convert_cot_to_sft.py \
    --input datasets/DDI13/cot_data_get/output/final_cot_20251219.jsonl \
    --output datasets/DDI13/sft_cot_data/train.jsonl

# Convert test set CoT data
python datasets/DDI13/cot_data_get/convert_cot_to_sft.py \
    --input datasets/DDI13/cot_data_get/output/final_cot_test.jsonl \
    --output datasets/DDI13/sft_cot_data/test.jsonl
"""
