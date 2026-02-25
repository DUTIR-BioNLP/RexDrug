#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDI13 Chain-of-Thought Data Generation Script

This script generates reasoning chains (Chain-of-Thought) for DDI13 dataset
using OpenAI API with multi-threaded processing.

Features:
- Multi-threaded API calls for faster processing
- Automatic retry with evaluation feedback
- Quality evaluation for each generated reasoning chain
- Progress tracking with detailed statistics

Input format (JSONL):
{
    "id": 0,
    "context": "sentence text...",
    "entities": [{"entity": "drug1", "category": "drug"}, ...],
    "relations": [{"type": "effect", "ent_sets": ["drug1", "drug2"]}, ...]
}

Output format (JSONL):
{
    "idx": 1,
    "sentence": "...",
    "gold_re": [{"type": "effect", "ent_set": ["drug1", "drug2"]}],
    "think": "reasoning content",
    "answer": "[{...}]",
    "scores": {...}
}

Usage:
    python get_cot_data.py --input ../DDI2013/train.jsonl --output ./output
"""

import os
import json
import threading
import re
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    raise

# Import prompts from local module
from prompt import COT_TEACHER_PROMPT, EVALUATION_PROMPT


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON object or array from model output text.
    Handles nested brackets and various formatting issues.

    Args:
        text: Raw text from model output

    Returns:
        Parsed JSON object/array or None if extraction fails
    """
    if not isinstance(text, str):
        return None

    # Try direct parsing first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the first bracket and extract JSON block
    start_idx = text.find('[')
    brace_start = text.find('{')

    if start_idx == -1 and brace_start == -1:
        return None

    # Choose the first occurring bracket
    if start_idx == -1:
        start_idx = brace_start
        open_char, close_char = '{', '}'
    elif brace_start == -1:
        open_char, close_char = '[', ']'
    elif start_idx < brace_start:
        open_char, close_char = '[', ']'
    else:
        start_idx = brace_start
        open_char, close_char = '{', '}'

    # Match brackets to find complete JSON
    depth = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start_idx:i+1])
                except Exception as e:
                    print(f"JSON parse error: {e}")
                    return None
    return None


def extract_think_content(text: str) -> str:
    """Extract content from <think>...</think> tags."""
    match = re.search(r'<think>([\s\S]*?)</think>', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer_content(text: str) -> str:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r'<answer>([\s\S]*?)</answer>', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def save_jsonl(path: str, data: List[dict], mode: str = 'w') -> None:
    """Save list of dictionaries to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_relations_to_gold_re(relations: List[dict]) -> List[dict]:
    """
    Convert DDI13 relations format to gold_re format.

    Input format:
        {"type": "effect", "ent_sets": ["drug1", "drug2"]}

    Output format:
        {"type": "effect", "ent_set": ["drug1", "drug2"]}

    Args:
        relations: List of relation dictionaries

    Returns:
        List of gold_re formatted dictionaries
    """
    # Valid relation types (excluding 'other' which maps to NO_COMB)
    valid_types = {"effect", "advise", "int", "mechanism"}

    gold_re = []
    for rel in relations:
        rel_type = rel.get("type", "").lower().strip()
        ent_sets = rel.get("ent_sets", [])

        # Skip 'other' type - these are non-interactions
        if rel_type == "other" or rel_type not in valid_types:
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
            gold_re.append({
                "type": rel_type,
                "ent_set": ent_set
            })

    # If no valid relations, add NO_COMB
    if not gold_re:
        gold_re.append({
            "type": "NO_COMB",
            "ent_set": []
        })

    return gold_re


# =============================================================================
# API CLIENT CONFIGURATION
# =============================================================================

def create_client(api_key: str, base_url: str = None) -> OpenAI:
    """Create OpenAI client with given credentials."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )


# =============================================================================
# COT GENERATION AND EVALUATION
# =============================================================================

# Configuration
MAX_RETRIES = 3
MIN_ACCEPTABLE_SCORE = 4  # Minimum score per dimension to pass
SCORE_KEYS = [
    'score_format', 'score_medical', 'score_semantic',
    'score_re_consistency', 'score_naturalness', 'score_usefulness'
]


def build_cot_prompt(sentence: str, gold_re_str: str,
                     failed_feedback: str = None) -> str:
    """
    Build the complete prompt for CoT generation.

    Args:
        sentence: Target sentence
        gold_re_str: JSON string of gold relations
        failed_feedback: Feedback from previous failed attempts

    Returns:
        Complete prompt string
    """
    prompt = COT_TEACHER_PROMPT.format(
        TARGET_SENTENCE=sentence,
        GOLD_RE=gold_re_str
    )

    if failed_feedback:
        retry_header = f"""Your previous attempt(s) had some issues that need to be addressed.

--- Previous Attempt Feedback ---
{failed_feedback}
---------------------------------

Please generate a new reasoning chain that addresses these issues.
Pay special attention to:
1. Format: Ensure <think>...</think> and <answer>...</answer> tags are properly used.
2. Structure: Include all four sections [1], [2], [3], [4] with bullet points.
3. Consistency: The final JSON in <answer> must exactly match the GOLD_RE.
4. Naturalness: Do NOT mention ground truth, gold answer, or any hint that you know the answer.
5. Usefulness: Provide clinically meaningful reasoning with textual evidence.

Now, please try again:

"""
        return retry_header + prompt

    return prompt


def generate_cot(client: OpenAI, model: str, sentence: str, gold_re: List[dict],
                 failed_attempts: Optional[List[dict]] = None) -> str:
    """
    Generate reasoning chain using API.

    Args:
        client: OpenAI client instance
        model: Model name to use
        sentence: Target sentence
        gold_re: Expert-annotated relations
        failed_attempts: Previous failed attempts with feedback

    Returns:
        Model output text containing <think> and <answer> sections
    """
    gold_re_str = json.dumps(gold_re, ensure_ascii=False, separators=(',', ':'))

    # Build feedback from failed attempts
    failed_feedback = None
    if failed_attempts:
        feedback_parts = []
        for i, attempt in enumerate(failed_attempts, 1):
            feedback_parts.append(
                f"Attempt {i}:\n"
                f"Issues: {attempt.get('comment', 'No comment')}\n"
                f"Scores: {attempt.get('scores_summary', 'N/A')}"
            )
        failed_feedback = "\n\n".join(feedback_parts)

    prompt = build_cot_prompt(sentence, gold_re_str, failed_feedback)

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )

    return completion.choices[0].message.content or ""


def evaluate_cot(client: OpenAI, model: str, sentence: str,
                 gold_re: List[dict], model_output: str) -> Dict[str, Any]:
    """
    Evaluate the quality of generated reasoning chain.

    Args:
        client: OpenAI client instance
        model: Model name for evaluation
        sentence: Target sentence
        gold_re: Expert-annotated relations
        model_output: Generated output with <think> and <answer>

    Returns:
        Dictionary with scores and comments
    """
    gold_re_str = json.dumps(gold_re, ensure_ascii=False, separators=(',', ':'))

    eval_input = f"""{EVALUATION_PROMPT}

--- Input for Evaluation ---
Target Sentence: {sentence}

GOLD_RE (Expert-validated):
{gold_re_str}

Model Output:
{model_output}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': eval_input}]
    )

    raw_output = completion.choices[0].message.content
    parsed = safe_extract_json(raw_output)

    # Fallback handling
    if not isinstance(parsed, dict):
        parsed = {}

    def to_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    return {
        "score_format":         to_int(parsed.get("score_format", 0)),
        "score_medical":        to_int(parsed.get("score_medical", 0)),
        "score_semantic":       to_int(parsed.get("score_semantic", 0)),
        "score_re_consistency": to_int(parsed.get("score_re_consistency", 0)),
        "score_naturalness":    to_int(parsed.get("score_naturalness", 0)),
        "score_usefulness":     to_int(parsed.get("score_usefulness", 0)),
        "comment":              parsed.get("comment", "Evaluation parsing failed"),
        "raw_output":           raw_output
    }


def check_pass(scores: Dict[str, Any]) -> bool:
    """Check if all evaluation criteria pass minimum threshold."""
    return all(scores.get(k, 0) >= MIN_ACCEPTABLE_SCORE for k in SCORE_KEYS)


def get_scores_summary(scores: Dict[str, Any]) -> str:
    """Generate summary string of scores."""
    parts = [f"{k}={scores.get(k, 0)}" for k in SCORE_KEYS]
    return ", ".join(parts)


# =============================================================================
# MULTI-THREADED PROCESSING
# =============================================================================

# Global progress tracking (thread-safe)
progress_bar = None
progress_lock = threading.Lock()
stats_lock = threading.Lock()
global_stats = {"passed": 0, "failed": 0}


def update_progress(passed: bool):
    """Thread-safe progress update."""
    with stats_lock:
        if passed:
            global_stats["passed"] += 1
        else:
            global_stats["failed"] += 1
        p, f = global_stats["passed"], global_stats["failed"]

    with progress_lock:
        if progress_bar is not None:
            progress_bar.set_postfix({"pass": p, "fail": f}, refresh=True)
            progress_bar.update(1)


class WorkerThread(threading.Thread):
    """Worker thread for processing data slices."""

    def __init__(self, func, items, thread_id: int, client: OpenAI,
                 gen_model: str, eval_model: str):
        super().__init__()
        self.func = func
        self.items = items
        self.thread_id = thread_id
        self.client = client
        self.gen_model = gen_model
        self.eval_model = eval_model
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.func(
                self.items, self.thread_id, self.client,
                self.gen_model, self.eval_model
            )
        except Exception as e:
            self.result = {
                "first_outputs": [],
                "final_outputs": [],
                "logs": [],
                "failed_records": []
            }
            self.error = e
            print(f"[Thread-{self.thread_id}] Error: {e}")

    def get_result(self):
        return self.result


def process_slice(items: List[tuple], thread_id: int,
                  client: OpenAI, gen_model: str,
                  eval_model: str) -> Dict[str, List[dict]]:
    """
    Process a slice of data items.

    Args:
        items: List of (idx, row) tuples
        thread_id: Thread identifier
        client: OpenAI client
        gen_model: Model name for CoT generation
        eval_model: Model name for CoT evaluation

    Returns:
        Dictionary with four result categories
    """
    first_outputs: List[dict] = []   # First generation attempts
    final_outputs: List[dict] = []   # Successfully passed outputs
    logs: List[dict] = []            # Detailed processing logs
    failed_records: List[dict] = []  # Failed records

    for idx, row in items:
        sentence = row.get('context', '')
        entities = row.get('entities', [])
        relations = row.get('relations', [])

        # Convert to gold_re format
        gold_re = convert_relations_to_gold_re(relations)

        retries = 0
        failed_attempts: List[dict] = []
        attempts_log: List[dict] = []
        last_output = ""
        passed = False

        while retries < MAX_RETRIES:
            try:
                # Generate reasoning chain
                output = generate_cot(
                    client, gen_model, sentence, gold_re,
                    failed_attempts if retries > 0 else None
                )
                last_output = output

                # Extract content
                think_content = extract_think_content(output)
                answer_content = extract_answer_content(output)

                # Evaluate
                scores = evaluate_cot(client, eval_model, sentence, gold_re, output)
                scores_summary = get_scores_summary(scores)

                # Record this attempt
                attempt_record = {
                    "retry": retries,
                    "output": output,
                    "think": think_content,
                    "answer": answer_content,
                    "scores": {k: scores.get(k, 0) for k in SCORE_KEYS},
                    "comment": scores.get("comment", ""),
                    "passed": check_pass(scores)
                }
                attempts_log.append(attempt_record)

                # Save first attempt
                if retries == 0:
                    first_outputs.append({
                        "idx": idx,
                        "sentence": sentence,
                        "entities": entities,
                        "original_relations": relations,
                        "gold_re": gold_re,
                        "output": output,
                        "think": think_content,
                        "answer": answer_content,
                        "scores": attempt_record["scores"],
                        "comment": scores.get("comment", "")
                    })

                # Check if passed
                if check_pass(scores):
                    final_outputs.append({
                        "idx": idx,
                        "sentence": sentence,
                        "entities": entities,
                        "original_relations": relations,
                        "gold_re": gold_re,
                        "output": output,
                        "think": think_content,
                        "answer": answer_content,
                        "scores": attempt_record["scores"],
                        "retry_count": retries
                    })
                    passed = True
                    break
                else:
                    # Record failure for retry
                    failed_attempts.append({
                        "output": output,
                        "comment": scores.get("comment", ""),
                        "scores_summary": scores_summary
                    })

                retries += 1

            except Exception as e:
                # Record exception
                failed_records.append({
                    "idx": idx,
                    "sentence": sentence,
                    "entities": entities,
                    "original_relations": relations,
                    "gold_re": gold_re,
                    "last_output": last_output,
                    "retry_count": retries,
                    "error": str(e),
                    "error_type": "exception"
                })
                break

        # If all retries failed (not due to exception)
        if not passed and retries >= MAX_RETRIES:
            best_attempt = max(
                attempts_log,
                key=lambda x: sum(x["scores"].values())
            ) if attempts_log else None

            failed_records.append({
                "idx": idx,
                "sentence": sentence,
                "entities": entities,
                "original_relations": relations,
                "gold_re": gold_re,
                "last_output": last_output,
                "best_output": best_attempt["output"] if best_attempt else "",
                "best_scores": best_attempt["scores"] if best_attempt else {},
                "retry_count": retries,
                "error_type": "max_retries_exceeded"
            })

        # Save log
        logs.append({
            "idx": idx,
            "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
            "gold_re": gold_re,
            "attempts": attempts_log,
            "final_status": "passed" if passed else "failed",
            "total_retries": retries
        })

        # Update progress
        update_progress(passed)

    return {
        "first_outputs": first_outputs,
        "final_outputs": final_outputs,
        "logs": logs,
        "failed_records": failed_records
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought data for DDI13 dataset"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input JSONL file path (e.g., ../DDI2013/train.jsonl)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./output",
        help="Output directory path (default: ./output)"
    )
    parser.add_argument(
        "--api-key", type=str, required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="API base URL (optional, for custom endpoints)"
    )
    parser.add_argument(
        "--gen-model", type=str, default="gpt-4o",
        help="Model name for CoT generation (default: gpt-4o)"
    )
    parser.add_argument(
        "--eval-model", type=str, default="gpt-5.1",
        help="Model name for CoT quality evaluation (default: gpt-5.1)"
    )
    parser.add_argument(
        "--threads", type=int, default=10,
        help="Number of parallel threads (default: 10)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples to process (optional)"
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # Create client
    client = create_client(args.api_key, args.base_url)

    # Output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_path = output_dir / f"first_cot_{timestamp}.jsonl"
    final_path = output_dir / f"final_cot_{timestamp}.jsonl"
    log_path = output_dir / f"generation_log_{timestamp}.jsonl"
    fail_path = output_dir / f"failed_records_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{timestamp}.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {input_path}")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Apply limit if specified
    if args.limit:
        data = data[:args.limit]

    indexed_items = list(enumerate(data, start=1))
    total = len(indexed_items)
    print(f"Total samples: {total}")

    # Statistics
    samples_with_rel = sum(
        1 for item in data
        if any(r.get("type", "").lower() not in ["other", ""]
               for r in item.get('relations', []))
    )
    samples_without_rel = total - samples_with_rel
    print(f"Samples with DDI relations: {samples_with_rel}")
    print(f"Samples without DDI relations (NO_COMB): {samples_without_rel}")

    # Thread configuration
    thread_count = min(args.threads, total)
    if thread_count < 1:
        thread_count = 1

    # Create slices
    slices = []
    slice_size = total // thread_count
    for i in range(thread_count):
        start = i * slice_size
        end = start + slice_size if i < thread_count - 1 else total
        slices.append(indexed_items[start:end])

    # Filter empty slices
    slices = [s for s in slices if s]

    print(f"Starting {len(slices)} threads...")
    start_time = datetime.now()

    # Initialize progress tracking
    global progress_bar, global_stats
    global_stats["passed"] = 0
    global_stats["failed"] = 0
    progress_bar = tqdm(
        total=total,
        desc="Processing",
        unit="item",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    # Start threads
    threads: List[WorkerThread] = []
    for i, part in enumerate(slices):
        t = WorkerThread(
            func=process_slice,
            items=part,
            thread_id=i,
            client=client,
            gen_model=args.gen_model,
            eval_model=args.eval_model
        )
        threads.append(t)
        t.start()

    # Wait and collect results
    all_first, all_final, all_logs, all_fails = [], [], [], []
    for t in threads:
        t.join()
        res = t.get_result() or {
            "first_outputs": [], "final_outputs": [],
            "logs": [], "failed_records": []
        }
        all_first.extend(res["first_outputs"])
        all_final.extend(res["final_outputs"])
        all_logs.extend(res["logs"])
        all_fails.extend(res["failed_records"])

    # Close progress bar
    progress_bar.close()
    progress_bar = None

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Sort by index
    all_first.sort(key=lambda x: x["idx"])
    all_final.sort(key=lambda x: x["idx"])
    all_logs.sort(key=lambda x: x["idx"])
    all_fails.sort(key=lambda x: x["idx"])

    # Print statistics
    print("\n" + "=" * 50)
    print(f"Generation completed in {duration:.2f} seconds")
    print(f"Total samples: {total}")
    print(f"First outputs (all first attempts): {len(all_first)}")
    print(f"Final outputs (passed): {len(all_final)}")
    print(f"Failed records: {len(all_fails)}")
    print(f"Pass rate: {len(all_final)/total*100:.2f}%")
    print("=" * 50)

    # Save results
    print("\nSaving results...")
    save_jsonl(str(first_path), all_first)
    save_jsonl(str(final_path), all_final)
    save_jsonl(str(log_path), all_logs)
    save_jsonl(str(fail_path), all_fails)

    # Save summary
    summary = {
        "run_timestamp": timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "gen_model": args.gen_model,
        "eval_model": args.eval_model,
        "total_samples": total,
        "samples_with_relations": samples_with_rel,
        "samples_without_relations": samples_without_rel,
        "thread_count": len(slices),
        "max_retries": MAX_RETRIES,
        "min_acceptable_score": MIN_ACCEPTABLE_SCORE,
        "duration_seconds": duration,
        "results": {
            "first_outputs": len(all_first),
            "final_outputs": len(all_final),
            "failed_records": len(all_fails),
            "pass_rate": f"{len(all_final)/total*100:.2f}%"
        },
        "output_files": {
            "first_cot": str(first_path),
            "final_cot": str(final_path),
            "log": str(log_path),
            "failed": str(fail_path)
        }
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - First CoT: {first_path}")
    print(f"  - Final CoT: {final_path}")
    print(f"  - Log: {log_path}")
    print(f"  - Failed: {fail_path}")
    print(f"  - Summary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()


"""
Example usage (run from RexDrug root directory):

# Generate CoT data for training set
python datasets/DDI13/cot_data_get/get_cot_data.py \
    --input datasets/DDI13/DDI2013/train.jsonl \
    --output datasets/DDI13/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 10

# Generate CoT data for test set with limit
python datasets/DDI13/cot_data_get/get_cot_data.py \
    --input datasets/DDI13/DDI2013/test.jsonl \
    --output datasets/DDI13/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 10 \
    --limit 100
"""
