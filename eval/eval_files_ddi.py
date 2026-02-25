import argparse
import json
import os
import re
import ast
from collections import defaultdict
from typing import List, Dict, Any, Tuple


# DDI relation types
ALLOWED_TYPES = {"effect", "advise", "int", "mechanism", "no_comb"}


def safe_eval(item):
    """
    Extract a list string of the form [{'type': ..., ...}, {...}] from text and safely convert it to an object.
    """
    if not isinstance(item, str):
        return item if isinstance(item, list) else []

    # Match the first [...] content starting with [
    match = re.search(r"(\[\s*\{.*?\}\s*\]|\[\s*\])", item, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return ast.literal_eval(json_str)
        except Exception:
            try:
                return json.loads(json_str)
            except Exception:
                return []
    return []


def parse_side(x):
    """General-purpose: convert a string or object to a Python object."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return safe_eval(x)
    return x


def norm_drug_name(name: str) -> str:
    """Normalize drug name."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def norm_rel(re_item: Dict) -> Tuple[str, Tuple[str, ...]]:
    """Normalize a relation: (type, sorted entity tuple)."""
    rel_type = re_item.get("type", "").lower().strip()
    ent_set = re_item.get("ent_set", [])
    if isinstance(ent_set, list):
        ent_set = tuple(sorted([norm_drug_name(e) for e in ent_set if isinstance(e, str)]))
    else:
        ent_set = ()
    return (rel_type, ent_set)


def is_no_comb(rel_list: List[Dict]) -> bool:
    """Determine whether this is a NO_COMB type."""
    if not rel_list:
        return True
    if len(rel_list) == 1:
        item = rel_list[0]
        rel_type = item.get("type", "").lower().strip()
        ent_set = item.get("ent_set", [])
        if rel_type == "no_comb" and (not ent_set or ent_set == []):
            return True
    return False


def line_to_pairs(line_data: Dict, doc_id: int) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Parse a single line of data and return gold, pred, error_stats.
    """
    err = {"structural_errors": 0, "relation_type_errors": 0, "ent_set_errors": 0}

    # Parse gold (labels)
    gold_raw = line_data.get("labels", line_data.get("label", "[]"))
    gold_list = parse_side(gold_raw) if gold_raw is not None else []
    if not isinstance(gold_list, list):
        gold_list = []

    gold = []
    for item in gold_list:
        if not isinstance(item, dict):
            continue
        rel_type = item.get("type", "").lower().strip()
        ent_set = item.get("ent_set", [])
        if rel_type:
            gold.append({"doc_id": doc_id, "type": rel_type, "ent_set": ent_set})

    # Parse pred (response)
    pred_raw = line_data.get("response", line_data.get("predict", "[]"))
    try:
        pred_list = parse_side(pred_raw)
        if not isinstance(pred_list, list):
            err["structural_errors"] += 1
            pred_list = []
    except Exception:
        err["structural_errors"] += 1
        pred_list = []

    pred = []
    for item in pred_list:
        if not isinstance(item, dict):
            err["structural_errors"] += 1
            continue

        rel_type = item.get("type", "").lower().strip()
        if rel_type not in ALLOWED_TYPES:
            err["relation_type_errors"] += 1
            continue

        ent_set = item.get("ent_set", [])
        if not isinstance(ent_set, list):
            err["ent_set_errors"] += 1
            continue

        # Non-NO_COMB types require at least 2 entities
        if rel_type != "no_comb" and len(ent_set) < 2:
            err["ent_set_errors"] += 1
            continue

        pred.append({"doc_id": doc_id, "type": rel_type, "ent_set": ent_set})

    return gold, pred, err


def compute_micro_f1(all_gold: List[Dict], all_pred: List[Dict]) -> Dict[str, float]:
    """
    Compute micro-averaged F1 (aggregate TP/FP/FN across all relation types).
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    # Group by doc_id
    gold_by_doc = defaultdict(list)
    pred_by_doc = defaultdict(list)

    for g in all_gold:
        gold_by_doc[g["doc_id"]].append(g)
    for p in all_pred:
        pred_by_doc[p["doc_id"]].append(p)

    all_doc_ids = set(gold_by_doc.keys()) | set(pred_by_doc.keys())

    for doc_id in all_doc_ids:
        gold_list = gold_by_doc.get(doc_id, [])
        pred_list = pred_by_doc.get(doc_id, [])

        # Handle the NO_COMB case
        gold_is_no_comb = is_no_comb(gold_list)
        pred_is_no_comb = is_no_comb(pred_list)

        if gold_is_no_comb and pred_is_no_comb:
            # Both are NO_COMB, count as one TP
            total_tp += 1
        elif gold_is_no_comb and not pred_is_no_comb:
            # Gold is NO_COMB, but pred predicted relations
            total_fp += len([p for p in pred_list if p["type"] != "no_comb"])
            total_fn += 1  # Missed the NO_COMB
        elif not gold_is_no_comb and pred_is_no_comb:
            # Gold has relations, but pred predicted NO_COMB
            total_fn += len([g for g in gold_list if g["type"] != "no_comb"])
            total_fp += 1  # Incorrectly predicted NO_COMB
        else:
            # Both have actual relations, perform set matching
            gold_rels = set(norm_rel(g) for g in gold_list if g["type"] != "no_comb")
            pred_rels = set(norm_rel(p) for p in pred_list if p["type"] != "no_comb")

            tp = len(gold_rels & pred_rels)
            fp = len(pred_rels - gold_rels)
            fn = len(gold_rels - pred_rels)

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"P": precision, "R": recall, "F1": f1}


def compute_macro_f1(all_gold: List[Dict], all_pred: List[Dict]) -> Dict[str, Any]:
    """
    Compute macro-averaged F1 (calculate F1 for each relation type separately, then average).
    """
    # Group by doc_id
    gold_by_doc = defaultdict(list)
    pred_by_doc = defaultdict(list)

    for g in all_gold:
        gold_by_doc[g["doc_id"]].append(g)
    for p in all_pred:
        pred_by_doc[p["doc_id"]].append(p)

    all_doc_ids = set(gold_by_doc.keys()) | set(pred_by_doc.keys())

    # Accumulate TP/FP/FN per type
    type_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for doc_id in all_doc_ids:
        gold_list = gold_by_doc.get(doc_id, [])
        pred_list = pred_by_doc.get(doc_id, [])

        gold_is_no_comb = is_no_comb(gold_list)
        pred_is_no_comb = is_no_comb(pred_list)

        if gold_is_no_comb and pred_is_no_comb:
            type_stats["no_comb"]["tp"] += 1
        elif gold_is_no_comb and not pred_is_no_comb:
            type_stats["no_comb"]["fn"] += 1
            for p in pred_list:
                if p["type"] != "no_comb":
                    type_stats[p["type"]]["fp"] += 1
        elif not gold_is_no_comb and pred_is_no_comb:
            type_stats["no_comb"]["fp"] += 1
            for g in gold_list:
                if g["type"] != "no_comb":
                    type_stats[g["type"]]["fn"] += 1
        else:
            # Group by type
            gold_by_type = defaultdict(set)
            pred_by_type = defaultdict(set)

            for g in gold_list:
                if g["type"] != "no_comb":
                    _, ent_tuple = norm_rel(g)
                    gold_by_type[g["type"]].add(ent_tuple)

            for p in pred_list:
                if p["type"] != "no_comb":
                    _, ent_tuple = norm_rel(p)
                    pred_by_type[p["type"]].add(ent_tuple)

            all_types = set(gold_by_type.keys()) | set(pred_by_type.keys())

            for rel_type in all_types:
                gold_ents = gold_by_type.get(rel_type, set())
                pred_ents = pred_by_type.get(rel_type, set())

                tp = len(gold_ents & pred_ents)
                fp = len(pred_ents - gold_ents)
                fn = len(gold_ents - pred_ents)

                type_stats[rel_type]["tp"] += tp
                type_stats[rel_type]["fp"] += fp
                type_stats[rel_type]["fn"] += fn

    # Compute F1 for each type
    type_f1 = {}
    f1_list = []

    for rel_type, stats in type_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        type_f1[rel_type] = {"P": precision, "R": recall, "F1": f1, "support": tp + fn}
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    return {"macro_F1": macro_f1, "per_type": type_f1}


def evaluate_jsonl(path: str) -> Dict[str, Any]:
    """Evaluate a single JSONL file."""
    all_gold, all_pred = [], []
    error_stats = {"structural_errors": 0, "relation_type_errors": 0, "ent_set_errors": 0}
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                data = json.loads(line)
            except Exception:
                error_stats["structural_errors"] += 1
                continue

            g, p, err = line_to_pairs(data, doc_id=i)
            all_gold.extend(g)
            all_pred.extend(p)
            for k in error_stats:
                error_stats[k] += err.get(k, 0)

    # Compute metrics
    micro_results = compute_micro_f1(all_gold, all_pred)
    macro_results = compute_macro_f1(all_gold, all_pred)

    return {
        "summary": {
            "total_lines": total_lines,
            "error_stats": error_stats
        },
        "micro": micro_results,
        "macro": {
            "F1": macro_results["macro_F1"],
            "per_type": macro_results["per_type"]
        }
    }


def extract_checkpoint_step(filename: str) -> int:
    """Extract the checkpoint step number from a filename."""
    match = re.search(r'checkpoint-(\d+)', filename)
    return int(match.group(1)) if match else 0


def evaluate_nested_directory(base_dir: str, output_path: str):
    """
    Traverse a nested directory structure: base_dir/seed_*/v*-*/checkpoint-*_results.jsonl
    """
    all_results = {}

    seed_dirs = sorted([d for d in os.listdir(base_dir)
                        if d.startswith("seed_") and os.path.isdir(os.path.join(base_dir, d))])

    if not seed_dirs:
        print(f"No seed directories found in {base_dir}")
        return

    for seed_dir in seed_dirs:
        seed_path = os.path.join(base_dir, seed_dir)
        seed_results = {}

        version_dirs = sorted([d for d in os.listdir(seed_path)
                               if os.path.isdir(os.path.join(seed_path, d))])

        for version_dir in version_dirs:
            version_path = os.path.join(seed_path, version_dir)
            version_results = {}

            checkpoint_files = sorted(
                [f for f in os.listdir(version_path) if f.endswith("_results.jsonl")],
                key=extract_checkpoint_step
            )

            for ckpt_file in checkpoint_files:
                ckpt_path = os.path.join(version_path, ckpt_file)
                print(f"Evaluating {seed_dir}/{version_dir}/{ckpt_file} ...")

                try:
                    metrics = evaluate_jsonl(ckpt_path)
                    ckpt_name = ckpt_file.replace("_results.jsonl", "")
                    version_results[ckpt_name] = metrics
                except Exception as e:
                    print(f"  Error evaluating {ckpt_file}: {e}")
                    continue

            if version_results:
                seed_results[version_dir] = version_results

        if seed_results:
            all_results[seed_dir] = seed_results

    # Save JSON results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {output_path}")

    # Print and save the summary table
    print_summary_table(all_results, output_path)


def generate_summary_table(results: Dict) -> str:
    """Generate a summary table string."""
    lines = []
    lines.append("=" * 110)
    lines.append("EVALUATION SUMMARY (DDI)")
    lines.append("=" * 110)
    lines.append(f"{'Seed':<12} {'Version':<25} {'Checkpoint':<20} {'micro_F1':<12} {'micro_P':<12} {'micro_R':<12} {'macro_F1':<12}")
    lines.append("-" * 110)

    for seed, versions in results.items():
        for version, checkpoints in versions.items():
            for ckpt, metrics in checkpoints.items():
                micro_f1 = metrics.get("micro", {}).get("F1", 0)
                micro_p = metrics.get("micro", {}).get("P", 0)
                micro_r = metrics.get("micro", {}).get("R", 0)
                macro_f1 = metrics.get("macro", {}).get("F1", 0)

                lines.append(f"{seed:<12} {version:<25} {ckpt:<20} {micro_f1:<12.4f} {micro_p:<12.4f} {micro_r:<12.4f} {macro_f1:<12.4f}")

    lines.append("=" * 110)
    return "\n".join(lines)


def print_summary_table(results: Dict, output_path: str = None):
    """Print the summary table and save it as a txt file."""
    table_str = generate_summary_table(results)
    print("\n" + table_str)

    if output_path:
        txt_path = output_path.replace(".json", "_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")
        print(f"\nSummary table saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDI relation extraction results")
    parser.add_argument("--data-dir", type=str,
                        required=True,
                        help="Directory containing inference results (.jsonl files)")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Path to save merged metrics JSON. Default: <data-dir>/evaluation_results.json")
    parser.add_argument("--mode", type=str, choices=["nested", "flat"], default="flat",
                        help="Directory mode: 'nested' for seed_*/v*-*/*.jsonl structure, 'flat' for direct *.jsonl files")
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), f"Not a directory: {args.data_dir}"

    if args.output is None:
        args.output = os.path.join(args.data_dir, "evaluation_results.json")

    if args.mode == "nested":
        evaluate_nested_directory(args.data_dir, args.output)
    else:
        # Flat mode
        results = {}
        files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".jsonl")])
        if not files:
            print("No JSONL files found in directory.")
            return

        for fname in files:
            fpath = os.path.join(args.data_dir, fname)
            print(f"Evaluating {fname} ...")
            metrics = evaluate_jsonl(fpath)
            results[fname] = metrics

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()

"""
Example usage:

# Flat mode (default): Evaluate all .jsonl files in a directory
python eval/eval_files_ddi.py \
  --data-dir outputs/inference \
  --output outputs/inference/evaluation_results.json \
  --mode flat

# Nested mode: For seed_*/v*-*/checkpoint-*_results.jsonl structure
python eval/eval_files_ddi.py \
  --data-dir outputs/inference/multi_seed_results \
  --output outputs/inference/multi_seed_results/evaluation_results.json \
  --mode nested
"""
