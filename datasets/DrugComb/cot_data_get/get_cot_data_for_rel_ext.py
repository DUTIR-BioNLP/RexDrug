import os
import json
import threading
import re
import argparse
from runpy import run_path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

DATASET_DIR = Path(__file__).resolve().parents[1]
_PROMPTS = run_path(DATASET_DIR / "prompt.py")
RE_COT_TEMPLATE = _PROMPTS["RE_COT_TEMPLATE"]
evaluation_prompt = _PROMPTS["evaluation_prompt"]

# =========================
# Utility Functions
# =========================

def safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from raw model text."""
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # Attempt to extract a JSON block (handles nested braces)
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    # Find the matching closing brace
    depth = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start_idx:i+1])
                except Exception as e:
                    print("JSON parse error:", e)
                    return None
    return None


def extract_think_content(text: str) -> str:
    """Extract content from <think>...</think> tags in model output."""
    match = re.search(r'<think>([\s\S]*?)</think>', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer_content(text: str) -> str:
    """Extract content from <answer>...</answer> tags in model output."""
    match = re.search(r'<answer>([\s\S]*?)</answer>', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def save_jsonl(path: str, data: List[dict], mode: str = 'w') -> None:
    """Write a list of records to a JSONL file.

    Args:
        path: Output file path.
        data: List of dictionaries to write.
        mode: Write mode, 'w' for overwrite (default), 'a' for append.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def ensure_parsed(data: Any) -> Any:
    """Ensure data is a parsed Python object, not a JSON string.

    If the input is a string that can be parsed as JSON, return the parsed
    result; otherwise return the input as-is.
    """
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


# =========================
# Client & Prompts
# =========================

def create_client(api_key: str, base_url: str = None) -> OpenAI:
    """Create OpenAI client with given credentials."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )

def build_cot_prompt(sentence: str, paragraph: str, gold_re_str: str,
                     failed_feedback: str = None) -> str:
    """Build the full CoT generation prompt, following the standard template."""
    # Fill in the template placeholders
    prompt = RE_COT_TEMPLATE.format(
        TARGET_SENTENCE=sentence,
        REFERENCE_CONTEXT=paragraph,
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
5. Reasoning quality: Provide clinically meaningful reasoning with textual evidence.

Now, please try again:

"""
        return retry_header + prompt
    else:
        return prompt

# =========================
# Core Functions
# =========================

MAX_RETRIES = 3
MIN_ACCEPTABLE_SCORE = 4  # Minimum score per criterion to pass
SCORE_KEYS = ['score_format', 'score_extraction', 'score_semantic',
              'score_medical', 'score_naturalness', 'score_reasoning']


def generate_cot(client: OpenAI, model: str, sentence: str, paragraph: str, gold_re: List[dict],
                 failed_attempts: Optional[List[dict]] = None) -> str:
    """Generate a chain-of-thought reasoning chain.

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        sentence: Target sentence.
        paragraph: Reference context.
        gold_re: Expert-validated relations.
        failed_attempts: Previous failed attempts (containing output and feedback).

    Returns:
        Full text output from the model.
    """
    gold_re_str = json.dumps(gold_re, ensure_ascii=False, separators=(',', ':'))

    # Build failure feedback if available
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

    # Build the prompt
    prompt = build_cot_prompt(sentence, paragraph, gold_re_str, failed_feedback)

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )

    message = completion.choices[0].message
    output = message.content or ""
    return output


def evaluate_cot(client: OpenAI, model: str, sentence: str, paragraph: str, gold_re: List[dict],
                 model_output: str) -> Dict[str, Any]:
    """Evaluate the quality of a chain-of-thought reasoning chain.

    Args:
        client: OpenAI client instance.
        model: Model name for evaluation.
        sentence: Target sentence.
        paragraph: Reference context.
        gold_re: Expert-validated relations.
        model_output: Model output (containing <think> and <answer>).

    Returns:
        Evaluation result dictionary.
    """
    gold_re_str = json.dumps(gold_re, ensure_ascii=False, separators=(',', ':'))

    eval_input = f"""{evaluation_prompt}

--- Input for Evaluation ---
Target Sentence: {sentence}

Reference Context: {paragraph}

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

    result = {
        "score_format":      to_int(parsed.get("score_format", 0)),
        "score_extraction":  to_int(parsed.get("score_extraction", 0)),
        "score_semantic":    to_int(parsed.get("score_semantic", 0)),
        "score_medical":     to_int(parsed.get("score_medical", 0)),
        "score_naturalness": to_int(parsed.get("score_naturalness", 0)),
        "score_reasoning":   to_int(parsed.get("score_reasoning", 0)),
        "comment":           parsed.get("comment", "Evaluation parsing failed"),
        "raw_output":        raw_output  # Keep raw output for debugging
    }

    return result


def check_pass(scores: Dict[str, Any]) -> bool:
    """Check whether all evaluation criteria are met."""
    return all(scores.get(k, 0) >= MIN_ACCEPTABLE_SCORE for k in SCORE_KEYS)


def get_scores_summary(scores: Dict[str, Any]) -> str:
    """Generate a summary string of scores."""
    parts = [f"{k}={scores.get(k, 0)}" for k in SCORE_KEYS]
    return ", ".join(parts)


# =========================
# Multi-threaded Processing
# =========================

# Global progress bar and statistics (thread-safe)
progress_bar = None  # type: Optional[tqdm]
progress_lock = threading.Lock()
stats_lock = threading.Lock()
global_stats = {"passed": 0, "failed": 0}


def update_progress(passed: bool):
    """Update progress bar and statistics in a thread-safe manner."""
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
    """Worker thread that processes a slice of data."""

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
            self.result = self.func(self.items, self.thread_id, self.client,
                                    self.gen_model, self.eval_model)
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
    """Process a slice of data items.

    Args:
        items: [(idx, row), ...] list of indexed data items.
        thread_id: Thread ID.
        client: OpenAI client instance.
        gen_model: Model name for CoT generation.
        eval_model: Model name for CoT evaluation.

    Returns:
        Dictionary containing four categories of results.
    """
    first_outputs: List[dict] = []   # First-attempt CoT outputs
    final_outputs: List[dict] = []   # Final passed CoT outputs
    logs: List[dict] = []            # Detailed generation logs
    failed_records: List[dict] = []  # Failed records

    for idx, row in items:
        sentence = row['sentence']
        paragraph = row['paragraph']
        # Ensure rels and entities are parsed objects, avoiding double JSON escaping
        gold_re = ensure_parsed(row['rels'])
        entities = ensure_parsed(row.get('entities', []))

        retries = 0
        failed_attempts: List[dict] = []
        attempts_log: List[dict] = []
        last_output = ""
        passed = False

        while retries < MAX_RETRIES:
            try:
                # Generate CoT reasoning chain
                output = generate_cot(client, gen_model, sentence, paragraph, gold_re,
                                      failed_attempts if retries > 0 else None)
                last_output = output

                # Extract content
                think_content = extract_think_content(output)
                answer_content = extract_answer_content(output)

                # Evaluate
                scores = evaluate_cot(client, eval_model, sentence, paragraph, gold_re, output)
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

                # Save first attempt to first_outputs
                if retries == 0:
                    first_outputs.append({
                        "idx": idx,
                        "sentence": sentence,
                        "paragraph": paragraph,
                        "entities": entities,
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
                        "paragraph": paragraph,
                        "entities": entities,
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
                    # Record failure for next retry
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
                    "paragraph": paragraph,
                    "entities": entities,
                    "gold_re": gold_re,
                    "last_output": last_output,
                    "retry_count": retries,
                    "error": str(e),
                    "error_type": "exception"
                })
                break

        # If all retries exhausted (non-exception exit)
        if not passed and retries >= MAX_RETRIES:
            # Find the attempt with the highest total score
            best_attempt = max(attempts_log, key=lambda x: sum(x["scores"].values())) if attempts_log else None
            failed_records.append({
                "idx": idx,
                "sentence": sentence,
                "paragraph": paragraph,
                "entities": entities,
                "gold_re": gold_re,
                "last_output": last_output,
                "best_output": best_attempt["output"] if best_attempt else "",
                "best_scores": best_attempt["scores"] if best_attempt else {},
                "retry_count": retries,
                "error_type": "max_retries_exceeded"
            })

        # Save log entry
        logs.append({
            "idx": idx,
            "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
            "gold_re": gold_re,
            "attempts": attempts_log,
            "final_status": "passed" if passed else "failed",
            "total_retries": retries
        })

        # Update global progress bar
        update_progress(passed)

    return {
        "first_outputs": first_outputs,
        "final_outputs": final_outputs,
        "logs": logs,
        "failed_records": failed_records
    }


# =========================
# Main Entry Point
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Chain-of-Thought data for DrugComb dataset"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input JSON file path (e.g., ../ord_data/train.jsonl)"
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
        "--threads", type=int, default=50,
        help="Number of parallel threads (default: 50)"
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
    first_path = str(output_dir / f"test_first_cot_{timestamp}.jsonl")
    final_path = str(output_dir / f"test_final_cot_{timestamp}.jsonl")
    log_path = str(output_dir / f"test_generation_log_{timestamp}.jsonl")
    fail_path = str(output_dir / f"test_failed_records_{timestamp}.jsonl")
    summary_path = str(output_dir / f"test_summary_{timestamp}.json")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply limit if specified
    if args.limit:
        data = data[:args.limit]

    indexed_items = list(enumerate(data, start=1))
    total = len(indexed_items)
    print(f"Total samples: {total}")

    # Thread configuration
    thread_count = min(args.threads, total)
    if thread_count < 1:
        thread_count = 1

    # Evenly partition data into slices
    slices = []
    slice_size = total // thread_count
    for i in range(thread_count):
        start = i * slice_size
        end = start + slice_size if i < thread_count - 1 else total
        slices.append(indexed_items[start:end])

    # Filter out empty slices
    slices = [s for s in slices if s]

    print(f"Starting {len(slices)} threads...")
    start_time = datetime.now()

    # Initialize global progress bar and statistics (reset)
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

    # Launch threads
    threads: List[WorkerThread] = []
    for i, part in enumerate(slices):
        t = WorkerThread(func=process_slice, items=part, thread_id=i,
                         client=client, gen_model=args.gen_model,
                         eval_model=args.eval_model)
        threads.append(t)
        t.start()

    # Wait for all threads and aggregate results
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
    print("\n" + "="*50)
    print(f"Generation completed in {duration:.2f} seconds")
    print(f"Total samples: {total}")
    print(f"First outputs (all first attempts): {len(all_first)}")
    print(f"Final outputs (passed): {len(all_final)}")
    print(f"Failed records: {len(all_fails)}")
    print(f"Pass rate: {len(all_final)/total*100:.2f}%")
    print("="*50)

    # Save results
    print("\nSaving results...")
    save_jsonl(first_path, all_first)
    save_jsonl(final_path, all_final)
    save_jsonl(log_path, all_logs)
    save_jsonl(fail_path, all_fails)

    # Save run summary
    summary = {
        "run_timestamp": timestamp,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "gen_model": args.gen_model,
        "eval_model": args.eval_model,
        "total_samples": total,
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
            "first_cot": first_path,
            "final_cot": final_path,
            "log": log_path,
            "failed": fail_path
        }
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - First COT: {first_path}")
    print(f"  - Final COT: {final_path}")
    print(f"  - Log: {log_path}")
    print(f"  - Failed: {fail_path}")
    print(f"  - Summary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()


"""
Example usage (run from RexDrug root directory):

# Generate CoT data for training set
python datasets/DrugComb/cot_data_get/get_cot_data_for_rel_ext.py \
    --input datasets/DrugComb/ord_data/train.jsonl \
    --output datasets/DrugComb/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 50

# Generate CoT data for test set with limit
python datasets/DrugComb/cot_data_get/get_cot_data_for_rel_ext.py \
    --input datasets/DrugComb/ord_data/test.jsonl \
    --output datasets/DrugComb/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --gen-model gpt-4o \
    --eval-model gpt-5.1 \
    --threads 50 \
    --limit 100

# With custom API endpoint
python datasets/DrugComb/cot_data_get/get_cot_data_for_rel_ext.py \
    --input datasets/DrugComb/ord_data/train.jsonl \
    --output datasets/DrugComb/cot_data_get/output \
    --api-key YOUR_API_KEY \
    --base-url https://your-api-endpoint.com/v1 \
    --gen-model gpt-4o \
    --eval-model gpt-5.1
"""
