import argparse
import json
from enum import Enum
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-file', type=str, required=False, default="test_f1.jsonl", help="Path to the data file")
parser.add_argument('--output', type=str, required=False, default="test_f1_output.json", help="Path to the output metrics file")


class Label(Enum):
    NO_COMB = 0
    NEG = 1
    COMB = 1
    POS = 2


def get_label_pos_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 0, "COMB": 0, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 0, 0: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label


def get_label_any_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 1, "COMB": 1, "NO_COMB": 0}
    int_label2idx = {2: 1, 1: 1, 0: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label


def create_vectors(gold: List[Dict[str, Any]], test: List[Dict[str, Any]], exact_match: bool, any_comb: bool) \
        -> Tuple[Dict[Tuple[str, str, int], List[Tuple[int, float]]],
                 Dict[Tuple[str, str, int], List[Tuple[int, float]]]]:
    """This function constructs the gold and predicted vectors such that each gold/prediction,
        would be mapped to a list of its aligned counterparts. this alignment is needed for later metrics.

    Args:
        gold: a list of gold dictionaries each of which stands for a relation.
            each has a doc_id to identify which doc did it came from, drug_idxs to pinpoint the drugs participating in this relation,
            and a relation_label to state the gold labels.
        test: the same as gold but having the predicted labels instead.
        exact_match: if True, restricts the matching criteria to have the same spans in both relations.
            default is False, which gives the partial matching behavior in which we require at least two spans in common

    Example:
        gold: [{'doc_id': 1, 'drug_idxs': [1, 2], 'relation_label': 3}, {'doc_id': 2, 'drug_idxs': [0, 1], 'relation_label': 1}]
        test: [{'doc_id': 1, 'drug_idxs': [0, 1, 2], 'relation_label': 3}, {'doc_id': 2, 'drug_idxs': [0, 1], 'relation_label': 0}]
        unify negs: False
        exact match: False
        =>
        g_out: {(1, '[1, 2]', 3): [(3, 0.666)], (2, '[0, 1]', 1): [(0, 0)]}
        t_out: {(1, '[0, 1, 2]', 3): [(3, 0.666)]}

    Returns:
        gold and test dictionaries that map from each relation to its (partial/exact) matched labels and their scores
    """
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
            # 计算匹配的实体跨度
            spans_intersecting = len(set(rel1['drug_idxs']).intersection(set(rel2['drug_idxs'])))
            score = spans_intersecting / len(set(rel1['drug_idxs'] + rel2['drug_idxs']))
            # 如果我们有部分匹配（在允许的情况下）或精确匹配（在要求的情况下），则添加对齐关系。”
            if ((spans_intersecting >= 2) and (not exact_match)) or (score == 1):
                # 我们使用“行ID”（句子哈希、药物索引和标签）作为映射，并映射另一个向量的对齐关系列表（包括分数）。
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), score))
                found = True
                matched.add(k)
        # 如果测试没有发现金阳性，则添加假阴性对
        if not found:
            g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((Label.NO_COMB.value, 0))
    # 现在我们对测试中剩余的关系进行迭代，并添加假阳性
    for k, rel2 in enumerate(test):
        if k not in matched:
            t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((Label.NO_COMB.value, 0))
    return g_out, t_out


def get_max_sum_score(v, labeled):
    interesting = 0
    score = 0
    for (_, _, label), matched in v:
        if label != Label.NO_COMB.value:
            interesting += 1
            score += max([s if ((not labeled and other != Label.NO_COMB.value) or (other == label)) else 0 for other, s in matched])
    return score / interesting


def f_from_p_r(gs, ts, labeled=False):
    p = get_max_sum_score(ts.items(), labeled)
    r = get_max_sum_score(gs.items(), labeled)
    return (2 * p * r) / (p + r), p, r


def f_score(gold, test, exact_match=False, any_comb=False):
    gs, ts = create_vectors(gold, test, exact_match, any_comb=any_comb)
    f, p, r = f_from_p_r(gs, ts)
    return f, p, r


if __name__ == "__main__":

    args = parser.parse_args()
    
    with open(args.data_file) as f:
        data = [json.loads(l) for l in f.readlines()]
    for entry in data:
        entry['predict'] = entry['predict'].replace("'", '"')
        entry['label'] = entry['label'].replace("'", '"')

    gold  = []
    pred = []
    for ids, entry in enumerate(data):
        for item in json.loads(entry['label']):
            gold.append(
                {
                    "doc_id": ids,
                    "drug_idxs": item["ent_set"],
                    "relation_label": item["type"]
                }
            )
        for item in json.loads(entry["predict"]):
            pred.append(
                {
                    "doc_id": ids,
                    "drug_idxs": item["ent_set"],
                    "relation_label": item["type"]
                }
            )

    f_partial, p_partial, r_partial = f_score(gold, pred, exact_match=False, any_comb=True)
    f_labeled_partial, p_labeled_partial, r_labeled_partial = f_score(gold, pred, exact_match=False, any_comb=False)
    f_exact, p_exact, r_exact = f_score(gold, pred, exact_match=True, any_comb=True)
    f_labeled_exact, p_labeled_exact, r_labeled_exact = f_score(gold, pred, exact_match=True, any_comb=False)
    print(f"F1/P/R score: partial unlabeled = {f_partial, p_partial, r_partial}, partial labeled = {f_labeled_partial, p_labeled_partial, r_labeled_partial}")
    print(f"F1/P/R score: exact unlabeled = {f_exact, p_exact, r_exact}, exact labeled = {f_labeled_exact, p_labeled_exact, r_labeled_exact}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f_out:
        json.dump(
            {
                ""
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
