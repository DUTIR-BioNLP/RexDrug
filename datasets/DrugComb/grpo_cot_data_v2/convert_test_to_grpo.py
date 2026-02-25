#!/usr/bin/env python3
"""
Convert test set data to GRPO format (with empty <think> section).

Input format:  JSONL file, each line contains sentence, paragraph, entities, rels fields.
Output format: JSONL file, each line is {"messages": [{"role": "system", ...}, {"role": "user", ...}], "solution": "..."}
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

# System prompt
SYSTEM_PROMPT = (
    "You are an expert in biomedical drug-drug relation extraction.\n\n"
    "Your goal is to decide whether a sentence (with its context) contains any clinically relevant "
    "drug combination relations, and to explain your reasoning.\n\n"
    "Use these relation types:\n"
    "- POS: the drugs are used together or as a regimen and the text clearly describes a beneficial effect "
    "(e.g. better response, survival, or clear synergy).\n"
    "- NEG: the drugs are used together or interact and the text clearly describes a harmful effect "
    "(e.g. more toxicity or risk, reduced efficacy, or an explicit recommendation to avoid the combination).\n"
    "- COMB: the drugs are clearly given or considered together in the same regimen, patient, or experiment, "
    "but the overall effect is neutral, mixed, or not clearly judged as good or bad.\n"
    "- NO_COMB: multiple drugs are mentioned but only as lists, alternatives (A or B), different arms (A vs B), "
    "background or prior regimens, or purely theoretical/unproven interactions without clear real-world "
    "coadministration and outcome.\n\n"
    "Only predict POS/NEG/COMB when co-use or interaction in this setting is explicit or very strongly implied. "
    "If you are unsure, choose NO_COMB.\n\n"
    "When you respond, you must first think step by step inside <think>...</think> in a structured way, "
    "then output the final relations as a JSON array inside <answer>...</answer>."
)

# Input template
INPUT_TEMPLATE = (
    "Decide whether the following target sentence, using the context paragraph, contains any drug combination "
    "relations that should be extracted under the guideline in the system message.\n\n"
    "First, provide your reasoning inside <think>...</think> using FOUR sections:\n"
    "[1] Clinical scenario\n"
    "[2] Candidate drugs and regimen focus\n"
    "[3] Combination reasoning and clinical effect\n"
    "[4] Extraction-oriented clinical summary\n"
    "- Under each section, write only short bullet points starting with \"- \".\n"
    "- Keep the total reasoning concise, about 100–200 words.\n\n"
    "Then, immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.\n"
    "- Inside <answer>, return a valid JSON array with the form:\n"
    "<answer>\n"
    "[{{\"type\": \"POS|NEG|COMB|NO_COMB\", \"ent_set\": [\"drug1\", \"drug2\", \"...\"]}}]\n"
    "</answer>\n"
    "- Do not include any extra text or explanation outside <answer>.\n\n"
    "Target sentence: {sentence}\n"
    "Context paragraph: {paragraph}\n"
)


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


def convert_to_grpo_format(input_file: str, output_file: str) -> dict:
    """Convert test set data to GRPO format.

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
                user_content = INPUT_TEMPLATE.format(
                    sentence=sentence,
                    paragraph=paragraph
                )

                # Build answer
                answer = convert_rels_to_answer(rels)

                # Build solution (with empty <think> section)
                solution = f"<think>\n\n</think>\n\n<answer>\n{answer}\n</answer>"

                # Build GRPO format record
                grpo_data = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "solution": solution
                }

                # Write to output file
                f_out.write(json.dumps(grpo_data, ensure_ascii=False) + "\n")
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

    parser = argparse.ArgumentParser(description="Convert test data to GRPO format (empty think)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(script_dir / ".." / "ord_data" / "test.jsonl"),
        help="Input JSONL file path (raw test data with rels field)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(script_dir / "test_drugcomb.jsonl"),
        help="Output JSONL file path (default: test_drugcomb.jsonl in this directory)"
    )

    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        output_dir = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(output_dir / f"grpo_cot_drugcomb_test_{timestamp}.jsonl")

    print("=" * 60)
    print("Test Data to GRPO Converter (Empty Think)")
    print("=" * 60)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print("=" * 60)

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Run conversion
    stats = convert_to_grpo_format(args.input, args.output)

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
