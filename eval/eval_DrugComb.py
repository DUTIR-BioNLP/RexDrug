import asyncio
import json
import math
import re
from typing import Dict
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import argparse
import json
from enum import Enum
from collections import defaultdict,Counter
from typing import List, Dict, Any, Tuple
import os
import re
from difflib import SequenceMatcher
import ast


class Label(Enum):
    NO_COMB = 0
    NEG = 1
    COMB = 1
    POS = 2


def get_label_pos_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 0, "COMB": 0, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 0, 0: 0}
    if type(rel['relation_label']) == str:
        try:
            idx_label = str_label2idx[rel['relation_label']]
        except:
            idx_label = str_label2idx["NO_COMB"]
    else:
        try:
            idx_label = int_label2idx[rel['relation_label']]
        except:
            idx_label = int_label2idx[0]
    return idx_label


def get_label_any_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 2, "COMB": 2, "NO_COMB": 0}
    int_label2idx = {1: 2, 2: 1, 0: 0}
    if type(rel['relation_label']) == str:
        try:
            idx_label = str_label2idx[rel['relation_label']]
        except:
            idx_label = str_label2idx["NO_COMB"]
    else:
        try:
            idx_label = int_label2idx[rel['relation_label']]
        except:
            idx_label = int_label2idx[0]
    return idx_label


def create_vectors(gold: List[Dict[str, Any]], test: List[Dict[str, Any]], exact_match: bool, any_comb: bool) \
        -> Tuple[Dict[Tuple[str, str, int], List[Tuple[int, float]]],
                 Dict[Tuple[str, str, int], List[Tuple[int, float]]]]:

    g_out = defaultdict(list)
    t_out = defaultdict(list)
    if any_comb:
        get_label = get_label_any_comb
    else:
        get_label = get_label_pos_comb

    matched = set()
    for rel1 in gold:
        found = False
        for k, rel2 in enumerate(test):
            if rel1['doc_id'] != rel2['doc_id']:
                continue
            try:
                spans_intersecting = len(set(rel1['drug_idxs']).intersection(set(rel2['drug_idxs'])))
            
                score = spans_intersecting / (len(set(rel1['drug_idxs'] + rel2['drug_idxs'])) + 1e-19)
            except:
                spans_intersecting =0
                score = 0
            if ((spans_intersecting >= 2) and (not exact_match)) or (score >= 0.9999999999999):
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), score))
                found = True
                matched.add(k)
            elif (rel1['drug_idxs'] == [] and rel2['drug_idxs'] == []):
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
        # if label != Label.NO_COMB.value:
        interesting += 1
        score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0 for other, s in matched])
    return score / interesting


def f_from_p_r(gs, ts, labeled=False):
    p = get_max_sum_score(ts.items(), labeled)
    r = get_max_sum_score(gs.items(), labeled)
    return (2 * p * r) / (p + r + 1e-19), p, r


def f_score(gold, test, exact_match=False, any_comb=False):
    gs, ts = create_vectors(gold, test, exact_match, any_comb=any_comb)
    f, p, r = f_from_p_r(gs, ts)
    return f, p, r

def RE_f1(true_re, pred_re):
    gold = []
    pred = []
    for item in true_re:
        gold.append(
            {
                "doc_id": 1,
                "drug_idxs": item["ent_set"],
                "relation_label": item["type"]
            }
        )
    for item in pred_re:
        pred.append(
            {
                "doc_id": 1,
                "drug_idxs": item["ent_set"],
                "relation_label": item["type"]
            }
        )
    f_partial, p_partial, r_partial = f_score(gold, pred, exact_match=False, any_comb=True)
    f_exact, p_exact, r_exact = f_score(gold, pred, exact_match=True, any_comb=True)
    return f_partial, f_exact

def extract_re_method_2(text: str):
    try:
        re_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        re_list = eval(re_match.group(1).strip()) if re_match else []
    except:
        print("error")
        re_list = [{'type': 'NO_COMB', 'ent_set':["None"]}]
    
    return re_list


if __name__ == "__main__":
    input_dir = "your_output_directory" 

    input_file_names = os.listdir(input_dir)
    for file_name in input_file_names:
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(input_dir, 'eval_F1.jsonl')

        with open(input_file_path) as f:
            data = [json.loads(l) for l in f.readlines()]
    
        process_data = []
        for entry in data:
            process_data.append(
                {
                    'predict_re': extract_re_method_2(entry['predict']),
                    'label_re': ast.literal_eval(entry['label']),
                }
            )

        
        gold  = []
        pred = []
        for ids, entry in enumerate(process_data):
            # for item in json.loads(entry['label']):
            for item in entry['label_re']:
                gold.append(
                    {
                        "doc_id": ids,
                        "drug_idxs": item["ent_set"],
                        "relation_label": item["type"]
                    }
                )
            # for item in json.loads(entry["predict"]):
            try:
                for item in entry["predict_re"]:
                    for ent in item["ent_set"]:
                        type_error = False
                        if type(ent) == list:
                            type_error = True
                            
                    if not type_error:
                        pred.append(
                            {
                                "doc_id": ids,
                                "drug_idxs": item["ent_set"],
                                "relation_label": item["type"]
                            }
                        )
                    else:
                        pred.append(
                            {
                                "doc_id": ids,
                                "drug_idxs": [],
                                "relation_label": item["type"]
                            }
                        )
            except:
                pred.append(
                        {
                            "doc_id": ids,
                            "drug_idxs": [],
                            "relation_label": "NO_COMB"
                        }
                    )
        
        f_partial, p_partial, r_partial = f_score(gold, pred, exact_match=False, any_comb=True)
        f_labeled_partial, p_labeled_partial, r_labeled_partial = f_score(gold, pred, exact_match=False, any_comb=False)
        f_exact, p_exact, r_exact = f_score(gold, pred, exact_match=True, any_comb=True)
        f_labeled_exact, p_labeled_exact, r_labeled_exact = f_score(gold, pred, exact_match=True, any_comb=False)
        print(f"F1/P/R score: partial unlabeled = {f_partial, p_partial, r_partial}, partial labeled = {f_labeled_partial, p_labeled_partial, r_labeled_partial}")
        print(f"F1/P/R score: exact unlabeled = {f_exact, p_exact, r_exact}, exact labeled = {f_labeled_exact, p_labeled_exact, r_labeled_exact}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "a+") as f_out:
            json.dump(
                {
                    "The_target_experiment": file_name,
                    "f_ANY_partial": f_partial,
                    "p_ANY_partial": p_partial,
                    "r_ANY_partial": r_partial,
                    "f_POS_partial": f_labeled_partial,
                    "p_POS_partial": p_labeled_partial,
                    "r_POS_partial": r_labeled_partial,
                    "f_ANY_exact": f_exact,
                    "p_ANY_exact": p_exact,
                    "r_ANY_exact": r_exact,
                    "f_POS_exact": f_labeled_exact,
                    "p_POS_exact": p_labeled_exact,
                    "r_POS_exact": r_labeled_exact,
                }, f_out, indent=4)
            f_out.write("\n")