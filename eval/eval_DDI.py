import json
import os
from collections import defaultdict
from tqdm import tqdm
import re
import ast
def safe_eval(item):
    match = re.search(r"(\[\s*{.*?}\s*\])", item, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return ast.literal_eval(json_str)
        except Exception as e:
            print(f"error: {e}")
            return []
    else:
        return []

def RE_micro_F1_for_DDI(completions, solution, **kwargs) -> float:
    def norm_rel(re_item):
        rel_type = re_item.get("type", "").lower().strip()
        ent_set = tuple(sorted([e.lower().strip() for e in re_item.get("ent_set", [])]))
        return (rel_type, ent_set)

    def is_no_comb(rel_list):
        return len(rel_list) == 1 and rel_list[0].get("type", "").lower().strip() == "no_comb" and rel_list[0].get("ent_set") == []

    contents = [completion[0]["content"] for completion in completions]
    total_tp, total_fp, total_fn = 0, 0, 0

    for content, sol in zip(contents, solution):
        try:
            pred_re = safe_eval(content)
            true_re = safe_eval(sol)

            if is_no_comb(true_re) and is_no_comb(pred_re):
                total_tp += 1
            elif is_no_comb(true_re) != is_no_comb(pred_re):
                total_fp += 1 if not is_no_comb(pred_re) else 0
                total_fn += 1 if not is_no_comb(true_re) else 0
            else:
                true_rels = set(norm_rel(x) for x in true_re)
                pred_rels = set(norm_rel(x) for x in pred_re)
                tp = len(true_rels & pred_rels)
                fp = len(pred_rels - true_rels)
                fn = len(true_rels - pred_rels)

                total_tp += tp
                total_fp += fp
                total_fn += fn
        except:
            pass

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


input_dir = 'your_output_directory' 

input_file_names = os.listdir(input_dir)
input_file_names = sorted(input_file_names)
output_file_path = os.path.join(input_dir, 'eval_F1.jsonl')

results = []
for file_name in input_file_names:
    if file_name.startswith("eval"):
        continue
    input_file_path = os.path.join(input_dir, file_name)

    with open(input_file_path, encoding='utf-8') as f:
        data = [json.loads(l) for l in f.readlines()]

    completions = [[{'content': entry['predict']}] for entry in data]
    solution = [entry['label'] for entry in data]

    p,r,re_micro_f1 = RE_micro_F1_for_DDI(completions, solution)

    result = {
        "file": file_name,
        "p": p,
        "r": r,
        "RE_micro_F1": re_micro_f1
    }
    results.append(result)

with open(output_file_path, 'w', encoding='utf-8') as fw:
    for res in results:
        fw.write(json.dumps(res, ensure_ascii=False) + '\n')