import argparse
import json
import os
import re
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Any, Tuple


class Label(Enum):
    NO_COMB = 0
    NEG = 1
    COMB = 1
    POS = 2


ALLOWED_TYPES = {"POS", "NEG", "COMB", "NO_COMB"}


def get_label_pos_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 0, "COMB": 0, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 0, 0: 0}
    lab = rel['relation_label']
    if isinstance(lab, str):
        return str_label2idx[lab]
    return int_label2idx[lab]


def get_label_any_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 1, "COMB": 1, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 1, 0: 0}
    lab = rel['relation_label']
    if isinstance(lab, str):
        return str_label2idx[lab]
    return int_label2idx[lab]


def norm_drug_name(name: str) -> str:
    # Lowercase, strip leading/trailing whitespace, collapse spaces; removing surrounding punctuation can be enabled as needed (currently not removing alphanumeric characters and common symbols)
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_ent_set(ent_set: List[str]) -> Tuple[str, ...]:
    normed = []
    for x in ent_set:
        if isinstance(x, str):
            nx = norm_drug_name(x)
            if nx:
                normed.append(nx)
    # Deduplicate + sort
    normed = sorted(set(normed))
    return tuple(normed)


def create_vectors(gold: List[Dict[str, Any]], test: List[Dict[str, Any]], exact_match: bool, any_comb: bool):
    g_out = defaultdict(list)
    t_out = defaultdict(list)
    get_label = get_label_any_comb if any_comb else get_label_pos_comb
    matched = set()

    for i, rel1 in enumerate(gold):
        found = False
        for k, rel2 in enumerate(test):
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            set1 = set(rel1['drug_idxs'])
            set2 = set(rel2['drug_idxs'])
            inter = len(set1 & set2)
            union = len(set1 | set2) if (set1 or set2) else 1
            score = inter / (union + 1e-19)

            if ((inter >= 2 and not exact_match) or (score >= 0.9999999999)):
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), score))
                found = True
                matched.add(k)
            elif (len(set1) == 0 and len(set2) == 0):
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), 1-(1e-19)))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), 1-(1e-19)))
                found = True
                matched.add(k)
        if not found:
            g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((Label.NO_COMB.value, 0))
    for k, rel2 in enumerate(test):
        if k not in matched:
            t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((Label.NO_COMB.value, 0))
    return g_out, t_out


def get_max_sum_score(v, labeled):
    interesting = 0
    score = 0
    for (_, _, label), matched in v:
        interesting += 1
        score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0
                      for other, s in matched] or [0])
    return score / max(interesting, 1)


def f_from_p_r(gs, ts, labeled=False):
    p = get_max_sum_score(ts.items(), labeled)
    r = get_max_sum_score(gs.items(), labeled)
    return (2 * p * r) / (p + r + 1e-19), p, r


def f_score(gold, test, exact_match=False, any_comb=False):
    gs, ts = create_vectors(gold, test, exact_match, any_comb=any_comb)
    f, p, r = f_from_p_r(gs, ts, labeled=True)
    return f, p, r


def parse_side(x):
    """Generic: convert a string or object to a Python object"""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            # Some may contain escape sequences; retry with unicode decoding
            return json.loads(x.encode('utf-8').decode('unicode_escape'))
    return x


def line_to_pairs_with_errors(line_data, doc_id):
    """
    Returns:
      gold, pred, err_stats(dict):
        err_stats = {
          "structural_errors": int,
          "relation_type_errors": int,
          "ent_set_errors": int
        }
    """
    err = {"structural_errors": 0, "relation_type_errors": 0, "ent_set_errors": 0}

    # Treat gold data as trusted (if validation is needed, apply the same constraints to labels)
    # Supports "labels" (new format) or "label" (old format) field
    gold_raw = line_data.get("labels", line_data.get("label", "[]"))
    gold_list = parse_side(gold_raw) if gold_raw is not None else []
    if not isinstance(gold_list, list):
        # Extreme fallback: treat gold as empty if it is not a list
        gold_list = []

    gold: List[Dict[str, Any]] = []
    for item in gold_list:
        try:
            typ = str(item.get("type", "NO_COMB"))
            ents = normalize_ent_set(item.get("ent_set", []))
            gold.append({"doc_id": doc_id, "drug_idxs": ents, "relation_label": typ})
        except Exception:
            # Skip this gold entry if parsing fails
            continue

    # -- Key section: strict parsing and error classification for predictions -- #
    # Supports "response" (new format) or "predict" (old format) field
    pred: List[Dict[str, Any]] = []
    pred_raw = line_data.get("response", line_data.get("predict", "[]"))
    try:
        pred_list = parse_side(pred_raw)
        if not isinstance(pred_list, list):
            # Structural error: not a list
            err["structural_errors"] += 1
            pred_list = []
    except Exception:
        # Overall structural error: unable to parse as JSON
        err["structural_errors"] += 1
        pred_list = []

    for item in pred_list:
        # Each entry must be a dict
        if not isinstance(item, dict):
            err["structural_errors"] += 1
            continue

        # Check type
        t = item.get("type", None)
        if not isinstance(t, str) or t not in ALLOWED_TYPES:
            err["relation_type_errors"] += 1
            continue

        # Check ent_set
        ent_set = item.get("ent_set", None)
        if not isinstance(ent_set, list) or any(not isinstance(x, str) for x in ent_set):
            err["ent_set_errors"] += 1
            continue

        ents = normalize_ent_set(ent_set)
        # At least two drugs are required to count as a “combination”
        if t != "NO_COMB" and len(ents) < 2:
            err["ent_set_errors"] += 1
            continue

        pred.append({"doc_id": doc_id, "drug_idxs": ents, "relation_label": t})

    return gold, pred, err


def evaluate_jsonl(path: str):
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
            except Exception as e:
                # If even the outermost line is not valid JSON, count it as a structural error and treat this line as having no prediction
                error_stats["structural_errors"] += 1
                continue

            g, p, err = line_to_pairs_with_errors(data, doc_id=i)
            all_gold.extend(g)
            all_pred.extend(p)
            # Accumulate error statistics
            for k in error_stats:
                error_stats[k] += err.get(k, 0)

    # Metric computation
    f_partial_pos, p_partial_pos, r_partial_pos = f_score(all_gold, all_pred, exact_match=False, any_comb=False)
    f_exact_pos, p_exact_pos, r_exact_pos = f_score(all_gold, all_pred, exact_match=True, any_comb=False)
    f_partial_any, p_partial_any, r_partial_any = f_score(all_gold, all_pred, exact_match=False, any_comb=True)
    f_exact_any, p_exact_any, r_exact_any = f_score(all_gold, all_pred, exact_match=True, any_comb=True)

    return {
        "summary": {
            "total_lines": total_lines,
            "error_stats": error_stats
        },
        "pos_vs_other": {
            "partial": {"F1": f_partial_pos, "P": p_partial_pos, "R": r_partial_pos},
            "exact":   {"F1": f_exact_pos,   "P": p_exact_pos,   "R": r_exact_pos}
        },
        "any_comb": {
            "partial": {"F1": f_partial_any, "P": p_partial_any, "R": r_partial_any},
            "exact":   {"F1": f_exact_any,   "P": p_exact_any,   "R": r_exact_any}
        }
    }


def extract_checkpoint_step(filename: str) -> int:
    """Extract the checkpoint step number from the filename for sorting"""
    match = re.search(r'checkpoint-(\d+)', filename)
    return int(match.group(1)) if match else 0


def evaluate_nested_directory(base_dir: str, output_path: str):
    """
    Traverse the nested directory structure: base_dir/seed_*/v*-*/checkpoint-*_results.jsonl
    Evaluate each checkpoint file and aggregate the results.
    """
    all_results = {}

    # Iterate over all seed directories
    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("seed_") and os.path.isdir(os.path.join(base_dir, d))])

    if not seed_dirs:
        print(f"No seed directories found in {base_dir}")
        return

    for seed_dir in seed_dirs:
        seed_path = os.path.join(base_dir, seed_dir)
        seed_results = {}

        # Iterate over version directories (v*-*) under each seed
        version_dirs = sorted([d for d in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, d))])

        for version_dir in version_dirs:
            version_path = os.path.join(seed_path, version_dir)
            version_results = {}

            # Find all checkpoint result files
            checkpoint_files = sorted(
                [f for f in os.listdir(version_path) if f.endswith("_results.jsonl")],
                key=extract_checkpoint_step
            )

            for ckpt_file in checkpoint_files:
                ckpt_path = os.path.join(version_path, ckpt_file)
                print(f"Evaluating {seed_dir}/{version_dir}/{ckpt_file} ...")

                try:
                    metrics = evaluate_jsonl(ckpt_path)
                    # Extract checkpoint name (remove _results.jsonl suffix)
                    ckpt_name = ckpt_file.replace("_results.jsonl", "")
                    version_results[ckpt_name] = metrics
                except Exception as e:
                    print(f"  Error evaluating {ckpt_file}: {e}")
                    continue

            if version_results:
                seed_results[version_dir] = version_results

        if seed_results:
            all_results[seed_dir] = seed_results

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {output_path}")

    # Print summary table and save as txt
    print_summary_table(all_results, output_path)


def generate_summary_table(results: Dict) -> str:
    """Generate the summary table as a string"""
    lines = []
    lines.append("=" * 100)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 100)
    lines.append(f"{'Seed':<12} {'Version':<25} {'Checkpoint':<18} {'F1(pos,exact)':<15} {'F1(pos,partial)':<16} {'F1(any,exact)':<15} {'F1(any,partial)':<16}")
    lines.append("-" * 100)

    for seed, versions in results.items():
        for version, checkpoints in versions.items():
            for ckpt, metrics in checkpoints.items():
                f1_pos_partial = metrics.get("pos_vs_other", {}).get("partial", {}).get("F1", 0)
                f1_pos_exact = metrics.get("pos_vs_other", {}).get("exact", {}).get("F1", 0)
                f1_any_partial = metrics.get("any_comb", {}).get("partial", {}).get("F1", 0)
                f1_any_exact = metrics.get("any_comb", {}).get("exact", {}).get("F1", 0)

                lines.append(f"{seed:<12} {version:<25} {ckpt:<18} {f1_pos_exact:<15.4f} {f1_pos_partial:<16.4f} {f1_any_exact:<15.4f} {f1_any_partial:<16.4f}")

    lines.append("=" * 100)
    return "\n".join(lines)


def print_summary_table(results: Dict, output_path: str = None):
    """Print a concise summary table, and optionally save it as a txt file"""
    table_str = generate_summary_table(results)
    print("\n" + table_str)

    # Save as txt file
    if output_path:
        txt_path = output_path.replace(".json", "_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")
        print(f"\nSummary table saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate drug combination relation extraction results")
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

    # Default output path
    if args.output is None:
        args.output = os.path.join(args.data_dir, "evaluation_results.json")

    if args.mode == "nested":
        evaluate_nested_directory(args.data_dir, args.output)
    else:
        # Original flat mode
        results = {}
        agg_errors = {"structural_errors": 0, "relation_type_errors": 0, "ent_set_errors": 0}
        agg_lines = 0

        files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".jsonl")])
        if not files:
            print("No JSONL files found in directory.")
            return

        for fname in files:
            fpath = os.path.join(args.data_dir, fname)
            print(f"Evaluating {fname} ...")
            metrics = evaluate_jsonl(fpath)
            results[fname] = metrics
            agg_lines += metrics["summary"]["total_lines"]
            for k in agg_errors:
                agg_errors[k] += metrics["summary"]["error_stats"].get(k, 0)

        results["_folder_summary"] = {
            "total_files": len(files),
            "total_lines": agg_lines,
            "error_stats_aggregated": agg_errors
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()

"""
Example usage:

# Flat mode (default): Evaluate all .jsonl files in a directory
python eval/eval_files_drugcomb.py \
  --data-dir outputs/inference \
  --output outputs/inference/evaluation_results.json \
  --mode flat

# Nested mode: For seed_*/v*-*/checkpoint-*_results.jsonl structure
python eval/eval_files_drugcomb.py \
  --data-dir outputs/inference/multi_seed_results \
  --output outputs/inference/multi_seed_results/evaluation_results.json \
  --mode nested
"""