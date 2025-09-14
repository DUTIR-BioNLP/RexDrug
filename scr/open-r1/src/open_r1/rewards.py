"""Reward functions for GRPO training."""
"""
We will organize the code in the future to make its structure clearer and reduce redundancy.
"""
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
from .utils import is_e2b_available
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
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
    return idx_label


def get_label_any_comb(rel):
    str_label2idx = {"POS": 1, "NEG": 2, "COMB": 2, "NO_COMB": 0}
    int_label2idx = {1: 2, 2: 1, 0: 0}
    if type(rel['relation_label']) == str:
        idx_label = str_label2idx[rel['relation_label']]
    else:
        idx_label = int_label2idx[rel['relation_label']]
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
            # 计算匹配的实体跨度
            spans_intersecting = len(set(rel1['drug_idxs']).intersection(set(rel2['drug_idxs'])))
            score = spans_intersecting / (len(set(rel1['drug_idxs'] + rel2['drug_idxs'])) + 1e-19)
            # 如果我们有部分匹配（在允许的情况下）或精确匹配（在要求的情况下），则添加对齐关系。”
            if ((spans_intersecting >= 2) and (not exact_match)) or (score >= 0.9999999999999):
                # 我们使用“行ID”（句子哈希、药物索引和标签）作为映射，并映射另一个向量的对齐关系列表（包括分数）。
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), score))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), score))
                found = True
                matched.add(k)
            elif (rel1['drug_idxs'] == [] and rel2['drug_idxs'] == []):
                g_out[(rel1["doc_id"], str(rel1["drug_idxs"]), get_label(rel1))].append((get_label(rel2), 1-(1e-19)))
                t_out[(rel2["doc_id"], str(rel2["drug_idxs"]), get_label(rel2))].append((get_label(rel1), 1-(1e-19)))
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

def NER_process(true_ner, pred_ner):
    gold = []
    pred = []

    for item in true_ner:
        gold.append(item["ent"])
    try: 
        for item in pred_ner:
            pred.append(item["ent"])
    except:
        pred = []

    return gold, pred


def extract_ner(text: str):
    try:
        ner_match = re.search(r"@ner#(.*?)#ner@", text)
        ner_list = eval(ner_match.group(1)) if ner_match else []
    except:
        ner_list = [{'type': 'DRUG', 'ent': []}]

    
    return ner_list

def extract_re(text: str):
    try:
        re_match = re.search(r"@re#(.*?)#re@", text)
        re_list = eval(re_match.group(1)) if re_match else []
    except:
        re_list = [{'type': 'NO_COMB', 'ent_set':[]}]
    
    return re_list

def extract_re_method_2(text: str):
    try:
        re_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        re_list = eval(re_match.group(1)) if re_match else []
    except:
        re_list = [{'type': 'NO_COMB', 'ent_set':[]}]
    
    return re_list



def compute_ner_f1(true_labels, pred_labels) -> float:
    """计算ner F1分数"""
    true_set = set(true_labels[0])
    pred_set = set(pred_labels[0])
    
    if not true_set and not pred_set:
        return 1.0  # 两个都是空的，得分1
    if not true_set or not pred_set:
        return 0.0  # 其中一个为空，得分0
    
    tp = len(true_set & pred_set)  # 交集
    fp = len(pred_set - true_set)  # 假阳性
    fn = len(true_set - pred_set)  # 假阴性
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def new_NER_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_ner = extract_ner(sol)
            pred_ner = extract_ner(content)
            true_label, pred = NER_process(true_ner, pred_ner)
            ner_f1 = compute_ner_f1(true_label, pred)
            reward = ner_f1
            rewards.append(reward)
        except Exception as e:
            ner_f1 = 0
            reward = ner_f1
            rewards.append(reward)


    return rewards


def new_RE_accuracy_reward_with_partial_exact(completions, solution, **kwargs):
    """奖励函数：计算NER和RE的F1值"""
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = extract_re(sol)
            pred_re = extract_re(content)
            re_f_partial, re_f_exact = RE_f1(true_re, pred_re)

            # character_match_ratio = SequenceMatcher(None, true_re, pred_re).ratio()
            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)
        except Exception as e:
            re_f_partial = 0
            re_f_exact = 0

            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)

    return rewards

def new_RE_accuracy_reward_only_exact(completions, solution, **kwargs):
    """奖励函数：计算NER和RE的F1值"""
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = extract_re(sol)
            pred_re = extract_re(content)
            re_f_exact = RE_f1(true_re, pred_re)

            # character_match_ratio = SequenceMatcher(None, true_re, pred_re).ratio()
            reward = re_f_exact  # 平均奖励
            rewards.append(reward)
        except Exception as e:
            re_f_exact = 0

            reward = re_f_exact  # 平均奖励
            rewards.append(reward)

    return rewards

def RE_accuracy_reward_with_partial_exact_method_2(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            re_f_partial, re_f_exact = RE_f1(true_re, pred_re)

            # character_match_ratio = SequenceMatcher(None, true_re, pred_re).ratio()
            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)
        except Exception as e:
            re_f_partial = 0
            re_f_exact = 0
    
            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    rewards = []
    
    pattern = r"^<think>\n.*?\n</think>\n\s*<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    for content in completion_contents:
        reward = 0
        if re.match(pattern, content, re.DOTALL | re.MULTILINE):
            reward += 0.3
            try:
                pred_ner = extract_ner(content)
                pred_re = extract_re(content)

                if isinstance(pred_ner, list):
                    reward_tak = True
                    for item in pred_ner:
                        # 判断每个 item 是否是字典，并包含 'type' 和 'ent' 键
                        if isinstance(item, dict) and 'type' in item and 'ent' in item:
                            continue
                        else:
                            reward_tak = False
                    if reward_tak == True:
                        reward +=0.3
                if isinstance(pred_re, list):
                    reward_tak = True
                    for item in pred_re:
                        # 判断每个 item 是否是字典，并包含 'type' 和 'ent_set' 键
                        if isinstance(item, dict) and 'type' in item and 'ent_set' in item:
                            for i in item['ent_set']:
                                if isinstance(i, list):
                                    reward_tak = False
                                else:
                                    continue
                        else:
                            reward_tak = False
                    if reward_tak == True:
                        reward +=0.4
                rewards.append(reward)

            except Exception as e:
                rewards.append(reward)
        else:
            rewards.append(reward)
            
    return rewards

def format_reward_method_2(completions, **kwargs):
    rewards = []
    pattern = r"^<think>\n.*?\n</think>\n\s*<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    for content in completion_contents:
        reward = 0
        if re.match(pattern, content, re.DOTALL | re.MULTILINE):
            reward += 0.5
            try:
                pred_re = extract_re_method_2(content)

                if isinstance(pred_re, list):
                    reward_tak = True
                    for item in pred_re:
                        # 判断每个 item 是否是字典，并包含 'type' 和 'ent_set' 键
                        if isinstance(item, dict) and 'type' in item and 'ent_set' in item:
                            for i in item['ent_set']:
                                if isinstance(i, list):
                                    reward_tak = False
                                else:
                                    continue
                        else:
                            reward_tak = False
                    if reward_tak == True:
                        reward +=0.5
                rewards.append(reward)

            except Exception as e:
                rewards.append(reward)
        else:
            rewards.append(reward)
            
    return rewards

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        if text.count("<answer>\n") == 1:
            count += 0.125
        if text.count("\n</answer>") == 1:
            count += 0.125
        if text.count("@ner#") == 1:
            count += 0.125
        if text.count("#ner@") == 1:
            count += 0.125
        if text.count("@re#") == 1:
            count += 0.125
        if text.count("#re@") == 1:
            count += 0.125
        
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

def drug_combin_reward_with_penalty(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = extract_re(sol)
            pred_re = extract_re(content)
            the_drug_comb_pred = []
            the_true_comb = []
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.append(item['ent_set'])
                else:
                    the_drug_comb_pred.append(['No_ent_set'])

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.append(item['ent_set'])
                else:
                    the_true_comb.append(['No_ent_set'])
            
            all_overlap_score = 0
            for pred_set in the_drug_comb_pred:
                max_overlap_score = 0
                for true_set in the_true_comb:
                    score = len(set(true_set).intersection(set(pred_set))) / len(set(true_set).union(set(pred_set)))
                    max_overlap_score = max(max_overlap_score, score)
                all_overlap_score += max_overlap_score

            # 施加重复生成惩罚
            duplicate_count = 0
            pred_counts = Counter(tuple(sorted(sublist)) for sublist in the_drug_comb_pred)
            for count in pred_counts.values():
                if count > 1:
                    duplicate_count += count
            total_count = len(the_drug_comb_pred)
            repetition_ratio = duplicate_count / total_count
            penalty_duplicate = repetition_ratio * 1

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = all_overlap_score/len(the_drug_comb_pred) - penalty_duplicate - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards

def drug_combin_reward_with_penalty_method_2(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            the_drug_comb_pred = []
            the_true_comb = []
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.append(item['ent_set'])
                else:
                    the_drug_comb_pred.append(['No_ent_set'])

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.append(item['ent_set'])
                else:
                    the_true_comb.append(['No_ent_set'])
            
            all_overlap_score = 0
            for pred_set in the_drug_comb_pred:
                max_overlap_score = 0
                for true_set in the_true_comb:
                    score = len(set(true_set).intersection(set(pred_set))) / len(set(true_set).union(set(pred_set)))
                    max_overlap_score = max(max_overlap_score, score)
                all_overlap_score += max_overlap_score

            # 施加重复生成惩罚
            duplicate_count = 0
            pred_counts = Counter(tuple(sorted(sublist)) for sublist in the_drug_comb_pred)
            for count in pred_counts.values():
                if count > 1:
                    duplicate_count += count
            total_count = len(the_drug_comb_pred)
            repetition_ratio = duplicate_count / total_count
            penalty_duplicate = repetition_ratio * 1

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = all_overlap_score/len(the_drug_comb_pred) - penalty_duplicate - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards

def NER_F1_for_DDI(completions, solution, **kwargs) -> float:
    """
    true_ner_list/pred_ner_list: [{"type": "drug", "ent": ["A", "B"]}, ...]
    计算严格匹配的NER F1，实体类型不同视为不同实体
    """
    def get_all_entities(ner_list):
        ents = set()
        for item in ner_list:
            ent_type = item.get("type", "").lower().strip()
            for ent in item.get("ent", []):
                ents.add((ent_type, ent.lower().strip()))
        return ents

    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_ner = extract_ner(sol)
            pred_ner = extract_ner(content)
            true_ents = get_all_entities(true_ner)
            pred_ents = get_all_entities(pred_ner)

            tp = len(true_ents & pred_ents)
            fp = len(pred_ents - true_ents)
            fn = len(true_ents - pred_ents)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            rewards.append(f1)

        except Exception as e:
            ner_f1 = 0.0
            reward = ner_f1
            rewards.append(reward)

    return rewards

def RE_F1_for_DDI(completions, solution, **kwargs) -> float:
    """
    true_re_list/pred_re_list: [{"type": "mechanism", "ent_set": ["A", "B"]}, ...]
    计算严格匹配的RE F1：类型和实体集合均一致才算TP，顺序无关
    """
    def norm_rel(re_item):
        rel_type = re_item.get("type", "").lower().strip()
        ent_set = tuple(sorted([e.lower().strip() for e in re_item.get("ent_set", [])]))
        return (rel_type, ent_set)

    def is_no_comb(rel_list):
        return len(rel_list) == 1 and rel_list[0].get("type", "").lower().strip() == "no_comb" and rel_list[0].get("ent_set") == []
    
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = extract_re(sol)
            pred_re = extract_re(content)

            if is_no_comb(true_re) and is_no_comb(pred_re):
                rewards.append(1.0)
            elif is_no_comb(true_re) != is_no_comb(pred_re):
                rewards.append(0.0)
            else:
                true_rels = set(norm_rel(x) for x in true_re)
                pred_rels = set(norm_rel(x) for x in pred_re)
                tp = len(true_rels & pred_rels)
                fp = len(pred_rels - true_rels)
                fn = len(true_rels - pred_rels)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                rewards.append(f1)

        except Exception as e:
            reward = 0
            rewards.append(reward)
    return rewards

def entity_combin_reward_with_penalty_for_DDI(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = extract_re(sol)
            pred_re = extract_re(content)
            the_drug_comb_pred = set()
            the_true_comb = set()
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_drug_comb_pred.add(('No_ent_set'))

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_true_comb.add(('No_ent_set'))
            
           
            score = len(set(the_drug_comb_pred).intersection(set(the_true_comb))) / len(set(the_drug_comb_pred).union(set(the_true_comb)))
            

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = score - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards

def RE_F1_for_DDI_method_2(completions, solution, **kwargs) -> float:
    """
    true_re_list/pred_re_list: [{"type": "mechanism", "ent_set": ["A", "B"]}, ...]
    计算严格匹配的RE F1：类型和实体集合均一致才算TP，顺序无关
    """
    def norm_rel(re_item):
        rel_type = re_item.get("type", "").lower().strip()
        ent_set = tuple(sorted([e.lower().strip() for e in re_item.get("ent_set", [])]))
        return (rel_type, ent_set)

    def is_no_comb(rel_list):
        return len(rel_list) == 1 and rel_list[0].get("type", "").lower().strip() == "no_comb" and rel_list[0].get("ent_set") == []
    
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)

            if is_no_comb(true_re) and is_no_comb(pred_re):
                rewards.append(1.0)
            elif is_no_comb(true_re) != is_no_comb(pred_re):
                rewards.append(0.0)
            else:
                true_rels = set(norm_rel(x) for x in true_re)
                pred_rels = set(norm_rel(x) for x in pred_re)
                tp = len(true_rels & pred_rels)
                fp = len(pred_rels - true_rels)
                fn = len(true_rels - pred_rels)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                rewards.append(f1)

        except Exception as e:
            reward = 0
            rewards.append(reward)
    return rewards

def entity_combin_reward_with_penalty_for_DDI_method_2(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            the_drug_comb_pred = set()
            the_true_comb = set()
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_drug_comb_pred.add(('No_ent_set'))

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_true_comb.add(('No_ent_set'))
            
           
            score = len(set(the_drug_comb_pred).intersection(set(the_true_comb))) / len(set(the_drug_comb_pred).union(set(the_true_comb)))
            

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = score - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards

def reasoning_steps_reward(completions, solution, **kwargs):
    """
    Reward function that checks for clear step-by-step reasoning.
    Improved to match numbered lists and bullet points more robustly.
    """
    # 使用 (?mi) 来设置多行和忽略大小写
    pattern = (
        r"(?mi)"                         # 多行、忽略大小写
        r"(?:^step \d+:)"                # Step 1:
        r"|(?:^\d+\.)"                   # 1. 2. 3.
        r"|(?:^[-*]\s)"                  # - item  或 * item
        r"|(?:^first,|^second,|^next,|^finally,)" # Transition words at line start
    )

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]
    return [min(1.0, count / 3) for count in matches]

def drug_combin_reward_with_penalty_method_3_without_ner_task(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            the_drug_comb_pred = []
            the_true_comb = []
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.append(item['ent_set'])
                else:
                    the_drug_comb_pred.append(['No_ent_set'])

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.append(item['ent_set'])
                else:
                    the_true_comb.append(['No_ent_set'])
            
            all_overlap_score = 0
            for pred_set in the_drug_comb_pred:
                max_overlap_score = 0
                for true_set in the_true_comb:
                    score = len(set(true_set).intersection(set(pred_set))) / len(set(true_set).union(set(pred_set)))
                    max_overlap_score = max(max_overlap_score, score)
                all_overlap_score += max_overlap_score

            # 施加重复生成惩罚
            duplicate_count = 0
            pred_counts = Counter(tuple(sorted(sublist)) for sublist in the_drug_comb_pred)
            for count in pred_counts.values():
                if count > 1:
                    duplicate_count += count
            total_count = len(the_drug_comb_pred)
            repetition_ratio = duplicate_count / total_count
            penalty_duplicate = repetition_ratio * 1

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = all_overlap_score/len(the_drug_comb_pred) - penalty_duplicate - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards

def format_reward_method_3_without_ner_task(completions, **kwargs):
    rewards = []
    pattern = r"^<think>\n.*?\n</think>\n\s*<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    for content in completion_contents:
        reward = 0
        if re.match(pattern, content, re.DOTALL | re.MULTILINE):
            reward += 0.5
            try:
                pred_re = extract_re_method_2(content)

                if isinstance(pred_re, list):
                    reward_tak = True
                    for item in pred_re:
                        # 判断每个 item 是否是字典，并包含 'type' 和 'ent_set' 键
                        if isinstance(item, dict) and 'type' in item and 'ent_set' in item:
                            for i in item['ent_set']:
                                if isinstance(i, list):
                                    reward_tak = False
                                else:
                                    continue
                        else:
                            reward_tak = False
                    if reward_tak == True:
                        reward +=0.5
                rewards.append(reward)

            except Exception as e:
                rewards.append(reward)
        else:
            rewards.append(reward)
            
    return rewards

def RE_accuracy_reward_with_partial_exact_method_3_without_ner_task(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            re_f_partial, re_f_exact = RE_f1(true_re, pred_re)

            # character_match_ratio = SequenceMatcher(None, true_re, pred_re).ratio()
            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)
        except Exception as e:
            re_f_partial = 0
            re_f_exact = 0

            reward = re_f_partial*1/3 + re_f_exact*2/3  # 平均奖励
            rewards.append(reward)

    return rewards

def RE_F1_for_DDI_new_output(completions, solution, **kwargs) -> float:
    """
    true_re_list/pred_re_list: [{"type": "mechanism", "ent_set": ["A", "B"]}, ...]
    计算严格匹配的RE F1：类型和实体集合均一致才算TP，顺序无关
    """
    def norm_rel(re_item):
        rel_type = re_item.get("type", "").lower().strip()
        ent_set = tuple(sorted([e.lower().strip() for e in re_item.get("ent_set", [])]))
        return (rel_type, ent_set)

    def is_no_comb(rel_list):
        return len(rel_list) == 1 and rel_list[0].get("type", "").lower().strip() == "no_comb" and rel_list[0].get("ent_set") == []
    
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)

            if is_no_comb(true_re) and is_no_comb(pred_re):
                rewards.append(1.0)
            elif is_no_comb(true_re) != is_no_comb(pred_re):
                rewards.append(0.0)
            else:
                true_rels = set(norm_rel(x) for x in true_re)
                pred_rels = set(norm_rel(x) for x in pred_re)
                tp = len(true_rels & pred_rels)
                fp = len(pred_rels - true_rels)
                fn = len(true_rels - pred_rels)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                rewards.append(f1)

        except Exception as e:
            reward = 0
            rewards.append(reward)
    return rewards

def entity_combin_reward_with_penalty_new_output_for_DDI(completions, solution, **kwargs):
    # 将预测出的所有药物组合提取出来，按字母排序后，进行匹配，施加重复处罚
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, sol in zip(contents, solution):
        # 提取 NER 和 RE
        try:
            true_re = ast.literal_eval(sol)
            pred_re = extract_re_method_2(content)
            the_drug_comb_pred = set()
            the_true_comb = set()
            
            for item in pred_re:
                if item['ent_set'] != []:
                    the_drug_comb_pred.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_drug_comb_pred.add(('No_ent_set'))

            for item in true_re:
                if item['ent_set'] != []:
                    the_true_comb.add(tuple(sorted([e.lower().strip() for e in item.get("ent_set", [])])))
                else:
                    the_true_comb.add(('No_ent_set'))
            
           
            score = len(set(the_drug_comb_pred).intersection(set(the_true_comb))) / len(set(the_drug_comb_pred).union(set(the_true_comb)))
            

            # 施加错误空值惩罚
            The_Penalty = False
            for label in true_re:
                if label['type'] != 'NO_COMB':
                    for pred in pred_re:
                        if pred['type'] == 'NO_COMB' or pred['ent_set'] == []:
                            The_Penalty = True
            if  The_Penalty:
                penalty_error_NONE = 1
            else:
                penalty_error_NONE = 0

            reward = score - penalty_error_NONE
            rewards.append(reward)

        except Exception as e:
            reward = 0
            rewards.append(reward)

    return rewards


