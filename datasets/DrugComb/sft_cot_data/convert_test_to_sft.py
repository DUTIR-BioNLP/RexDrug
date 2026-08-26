#!/usr/bin/env python3
"""
Convert test set data to SFT instruction-tuning format (with empty <think> section).

Input format:  JSONL file, each line contains sentence, paragraph, entities, rels fields.
Output format: JSONL file, each line is {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

import json
import argparse
from runpy import run_path
from pathlib import Path
from datetime import datetime

DATASET_DIR = Path(__file__).resolve().parents[1]
_PROMPTS = run_path(DATASET_DIR / "prompt.py")
SYSTEM_PROMPT = _PROMPTS["SYSTEM_PROMPT"]
USER_PROMPT_TEMPLATE = _PROMPTS["USER_PROMPT_TEMPLATE"]


def convert_rels_to_answer(rels: list) -> str:
    """Convert the rels field to answer format.

    Args:
        rels: Raw relations list.

    Returns:
        JSON-formatted answer string.

    Note: The original data format uses {"class": "...", "entities": [...]}
          which is converted to {"type": "...", "ent_set": [...]}.
    """
    if not rels:
        # If no relations, return NO_COMB
        return json.dumps([{"type": "NO_COMB", "ent_set": []}], ensure_ascii=False)

    # Convert to standardized format
    # Original field mapping: class -> type, entities -> ent_set
    answer_list = []
    for rel in rels:
        # Prefer "type" field; fall back to "class"
        rel_type = rel.get("type", rel.get("class", "NO_COMB"))
        # Prefer "ent_set" field; fall back to "entities"
        ent_set = rel.get("ent_set", rel.get("entities", []))
        answer_list.append({"type": rel_type, "ent_set": ent_set})

    return json.dumps(answer_list, ensure_ascii=False)


def convert_to_sft_format(input_file: str, output_file: str) -> dict:
    """Convert test set data to SFT format.

    Args:
        input_file: Path to the input JSONL file.
        output_file: Path to the output JSONL file.

    Returns:
        A dictionary of conversion statistics.
    """
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "failed_indices": []
    }

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            stats["total"] += 1

            try:
                data = json.loads(line.strip())

                # Extract required fields
                sentence = data.get("sentence", "")
                paragraph = data.get("paragraph", "")
                rels = data.get("rels", [])

                # Validate required fields
                if not sentence or not paragraph:
                    stats["failed"] += 1
                    stats["failed_indices"].append(line_num)
                    print(f"Warning: Line {line_num} missing sentence or paragraph, skipping...")
                    continue

                # Build user input
                user_content = USER_PROMPT_TEMPLATE.format(
                    sentence=sentence,
                    paragraph=paragraph
                )

                # Build answer
                answer = convert_rels_to_answer(rels)

                # Build assistant output (with empty <think> section)
                assistant_content = f"<think>\n\n</think>\n\n<answer>\n{answer}\n</answer>"

                # Build SFT format record
                sft_data = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }

                # Write to output file
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


def main():
    # Resolve script directory for relative paths
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Convert test data to SFT instruction format (empty think)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(script_dir / ".." / "ord_data" / "test.jsonl"),
        help="Input JSONL file path (raw test data with rels field)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(script_dir / "drugcomb_test.jsonl"),
        help="Output JSONL file path (default: drugcomb_test.jsonl in this directory)"
    )

    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        output_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(output_dir / f"sft_cot_drugcomb_test_{timestamp}.jsonl")

    print("=" * 60)
    print("Test Data to SFT Converter (Empty Think)")
    print("=" * 60)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print("=" * 60)

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Run conversion
    stats = convert_to_sft_format(args.input, args.output)

    # Print statistics
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"Total samples:   {stats['total']}")
    print(f"Success:         {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    if stats['failed'] > 0:
        print(f"Failed indices:  {stats['failed_indices'][:10]}{'...' if len(stats['failed_indices']) > 10 else ''}")
    print(f"\nOutput saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
