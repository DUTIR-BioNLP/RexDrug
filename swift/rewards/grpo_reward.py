"""
GRPO Reward Function Plugin for DrugComb and DDI Datasets

DrugComb Dataset Reward Functions:
1. drugcomb_format: Format reward - Checks whether the output is in valid JSON format
2. drugcomb_coverage: Drug combination coverage reward - Computes maximum coverage of drug combinations + multiple penalty terms + duplicate penalty
3. drugcomb_accuracy: Accuracy reward - Computed based on Partial F1 and Exact F1

DrugComb COT (Chain-of-Thought) Reward Functions:
1. drugcomb_cot_format: COT format reward - Checks for <think> and <answer> tags and valid JSON format within the answer
2. drugcomb_cot_think: Thinking process reward - Checks whether <think> contains four numbered points [1][2][3][4]
3. drugcomb_coverage_cot: COT coverage reward - Extracts results from <answer> and computes coverage
4. drugcomb_accuracy_cot: COT accuracy reward - Extracts results from <answer> and computes accuracy

DDI Dataset Reward Functions:
1. ddi_format: Format reward - Checks whether the output is in valid JSON format
2. ddi_coverage: Drug combination coverage reward - Binary entity combination matching + NO_COMB penalty + duplicate penalty
3. ddi_accuracy: Accuracy reward - Micro-averaged F1 computation

DDI COT (Chain-of-Thought) Reward Functions:
1. ddi_cot_format: COT format reward - Checks for <think> and <answer> tags and valid JSON format within the answer
2. ddi_cot_think: Thinking process reward - Checks whether <think> contains four numbered points [1][2][3][4]
3. ddi_coverage_cot: COT coverage reward - Extracts results from <answer> and computes coverage
4. ddi_accuracy_cot: COT accuracy reward - Extracts results from <answer> and computes accuracy

Usage:
# DrugComb (standard mode)
swift rlhf \
    --external_plugins /path/to/plugin.py \
    --reward_funcs drugcomb_format drugcomb_coverage drugcomb_accuracy \
    --reward_weights 0.1 0.3 0.6

# DrugComb (COT chain-of-thought mode)
swift rlhf \
    --external_plugins /path/to/plugin.py \
    --reward_funcs drugcomb_cot_format drugcomb_cot_think drugcomb_coverage_cot drugcomb_accuracy_cot \
    --reward_weights 0.1 0.1 0.3 0.5

# DDI (standard mode)
swift rlhf \
    --external_plugins /path/to/plugin.py \
    --reward_funcs ddi_format ddi_coverage ddi_accuracy \
    --reward_weights 0.1 0.3 0.6

# DDI (COT chain-of-thought mode)
swift rlhf \
    --external_plugins /path/to/plugin.py \
    --reward_funcs ddi_cot_format ddi_cot_think ddi_coverage_cot ddi_accuracy_cot \
    --reward_weights 0.1 0.1 0.3 0.5
"""

import re
import json
import ast
from collections import Counter
from typing import List, Dict, Any, Tuple, Set
from swift.rewards import orms, ORM


# ==================== Common Utility Functions ====================

def safe_parse_json(text: str) -> List[Dict]:
    """
    Safely parse a JSON-formatted relation list.
    Supports both single-quote and double-quote formats.
    """
    if not isinstance(text, str):
        if isinstance(text, list):
            return text
        return []

    text = text.strip()
    if not text:
        return []

    # Try to match [...] format
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    json_str = match.group(0)

    # Try multiple parsing methods
    for parser in [json.loads, ast.literal_eval]:
        try:
            result = parser(json_str)
            if isinstance(result, list):
                return result
        except Exception:
            continue

    return []


def norm_drug_name(name: str) -> str:
    """Normalize drug name: lowercase, strip whitespace, collapse spaces"""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def normalize_ent_set(ent_set: List[str]) -> Tuple[str, ...]:
    """Normalize entity set: deduplicate, sort, and return as tuple"""
    if not isinstance(ent_set, list):
        return ()
    normed = []
    for x in ent_set:
        if isinstance(x, str):
            nx = norm_drug_name(x)
            if nx:
                normed.append(nx)
    return tuple(sorted(set(normed)))


def is_no_comb(rel_list: List[Dict]) -> bool:
    """Determine whether this is NO_COMB (no drug combination)"""
    if not rel_list:
        return True
    if len(rel_list) == 1:
        item = rel_list[0]
        if isinstance(item, dict):
            rel_type = str(item.get("type", "")).upper().strip()
            ent_set = item.get("ent_set", [])
            if rel_type == "NO_COMB" and (not ent_set or ent_set == []):
                return True
    return False


def get_rel_type(item: Dict) -> str:
    """Get the normalized relation type"""
    return str(item.get("type", "")).upper().strip()


def get_ent_set(item: Dict) -> List[str]:
    """Get the entity set"""
    ent_set = item.get("ent_set", [])
    if not isinstance(ent_set, list):
        return []
    return ent_set


def extract_relations(rel_list: List[Dict], min_ents: int = 2) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Extract normalized relations from the relation list.
    Returns: [(type, (drug1, drug2, ...)), ...]
    """
    relations = []
    if not isinstance(rel_list, list):
        return relations

    for item in rel_list:
        if not isinstance(item, dict):
            continue
        rel_type = get_rel_type(item)
        if rel_type == "NO_COMB":
            continue
        ent_set = normalize_ent_set(get_ent_set(item))
        if len(ent_set) >= min_ents:
            relations.append((rel_type, ent_set))

    return relations


def extract_drug_sets(rel_list: List[Dict], min_ents: int = 2) -> Set[Tuple[str, ...]]:
    """
    Extract only drug combination sets (ignoring relation types).
    Used for computing coverage.
    """
    drug_sets = set()
    if not isinstance(rel_list, list):
        return drug_sets

    for item in rel_list:
        if not isinstance(item, dict):
            continue
        rel_type = get_rel_type(item)
        if rel_type == "NO_COMB":
            continue
        ent_set = normalize_ent_set(get_ent_set(item))
        if len(ent_set) >= min_ents:
            drug_sets.add(ent_set)

    return drug_sets


# ==================== COT-Related Helper Functions ====================

def extract_cot_tags(text: str) -> Tuple[str, str]:
    """
    Extract <think> and <answer> tag contents from text.

    Returns:
        (think_content, answer_content): Contents of both tags; returns empty string if not present.
    """
    think_content = ""
    answer_content = ""

    # Extract <think>...</think> content
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()

    # Extract <answer>...</answer> content
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()

    return think_content, answer_content


def has_cot_tags(text: str) -> Tuple[bool, bool]:
    """
    Check whether the text contains <think> and <answer> tags.

    Returns:
        (has_think, has_answer): Whether each tag is present.
    """
    has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    return has_think, has_answer


def check_think_points(think_content: str) -> bool:
    """
    Check whether the thinking process contains four numbered points.

    Supported point formats:
    - [1] [2] [3] [4]
    - 1. 2. 3. 4.
    - 1、2、3、4、
    - 1) 2) 3) 4)
    - (1) (2) (3) (4)
    - 第一 第二 第三 第四 (Chinese: first, second, third, fourth)
    - 首先 其次 然后 最后 (Chinese: firstly, secondly, then, finally)

    Returns:
        bool: Whether all four points are present.
    """
    # Define multiple point formats
    point_patterns = [
        # [1] [2] [3] [4] format
        [r'\[1\]', r'\[2\]', r'\[3\]', r'\[4\]'],
        # 1. 2. 3. 4. format
        [r'1\.', r'2\.', r'3\.', r'4\.'],
        # 1、2、3、4、format (Chinese enumeration comma)
        [r'1、', r'2、', r'3、', r'4、'],
        # 1) 2) 3) 4) format
        [r'1\)', r'2\)', r'3\)', r'4\)'],
        # (1) (2) (3) (4) format
        [r'\(1\)', r'\(2\)', r'\(3\)', r'\(4\)'],
        # Chinese ordinal format: first, second, third, fourth
        ['第一', '第二', '第三', '第四'],
        # Chinese sequential format: firstly, secondly, then, finally
        ['首先', '其次', '然后', '最后'],
    ]

    for patterns in point_patterns:
        if all(re.search(p, think_content) for p in patterns):
            return True

    return False


def calculate_duplicate_penalty(pred_list: List[Dict], min_ents: int = 2) -> float:
    """
    Compute the penalty value for duplicate predictions.

    Args:
        pred_list: Predicted relation list
        min_ents: Minimum number of entities

    Returns:
        Penalty value (0.0 ~ 1.0)
    """
    if not isinstance(pred_list, list) or len(pred_list) == 0:
        return 0.0

    # Extract all valid drug combination tuples
    drug_comb_pred = []
    for item in pred_list:
        if not isinstance(item, dict):
            continue
        rel_type = get_rel_type(item)
        if rel_type == "NO_COMB":
            continue
        ent_set = get_ent_set(item)
        if len(ent_set) >= min_ents:
            drug_comb_pred.append(ent_set)

    if len(drug_comb_pred) == 0:
        return 0.0

    # Count duplicates
    duplicate_count = 0
    pred_counts = Counter(tuple(sorted(sublist)) for sublist in drug_comb_pred)
    for count in pred_counts.values():
        if count > 1:
            duplicate_count += count

    total_count = len(drug_comb_pred)
    repetition_ratio = duplicate_count / total_count
    penalty_duplicate = repetition_ratio * 1.0  # Penalty coefficient is 1

    return penalty_duplicate


# ==================== DrugComb Valid Relation Types ====================
DRUGCOMB_VALID_TYPES = {"POS", "NEG", "COMB", "NO_COMB"}

# ==================== DDI Valid Relation Types ====================
DDI_VALID_TYPES = {"EFFECT", "ADVISE", "INT", "MECHANISM", "NO_COMB"}


# ==================== Common Format Check Function ====================

def check_format(completion: str, valid_types: Set[str], min_ents: int = 2) -> float:
    """
    Check whether the output format is correct.

    Args:
        completion: Model output
        valid_types: Set of valid relation types
        min_ents: Minimum number of entities for non-NO_COMB relations

    Returns:
        1.0: Format is correct
        0.0: Format is incorrect
    """
    try:
        parsed = safe_parse_json(completion)

        if not isinstance(parsed, list):
            return 0.0

        if len(parsed) == 0:
            return 0.0

        for item in parsed:
            if not isinstance(item, dict):
                return 0.0

            rel_type = get_rel_type(item)
            if rel_type not in valid_types:
                return 0.0

            ent_set = item.get("ent_set")
            if not isinstance(ent_set, list):
                return 0.0

            # Non-NO_COMB types require at least min_ents entities
            if rel_type != "NO_COMB" and len(ent_set) < min_ents:
                return 0.0

        return 1.0

    except Exception:
        return 0.0


# ==================== DrugComb Reward Functions ====================

class DrugCombFormatORM(ORM):
    """
    DrugComb format reward: Checks whether the output is in valid JSON format.

    Reward rules:
    - 1.0: Output is a valid JSON list where each element contains type and ent_set fields
    - 0.0: Format is incorrect
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        return [check_format(c, DRUGCOMB_VALID_TYPES, min_ents=2) for c in completions]


class DrugCombCoverageORM(ORM):
    """
    DrugComb drug combination coverage reward: Computes maximum coverage of drug combinations + multiple penalty terms.

    Design purpose:
    - Compute maximum coverage of drug combinations (ignoring relation types)
    - Address reward sparsity issues

    Penalty terms (each deducts 1 point):
    1. False Negative: Ground truth has combinations but prediction is NO_COMB -> -1.0
    2. Empty Entity Set: Predicted type is not NO_COMB but entity set is empty -> -1.0
    3. Inconsistent NO_COMB: Predicted type is NO_COMB but entities were provided -> -1.0
    4. Duplicate Penalty: When predicted tuples have duplicates, deduct proportionally

    Coverage computation:
    - For each ground truth drug combination, find the predicted combination with the largest intersection
    - Coverage = |intersection| / |union| (Jaccard similarity)
    - Final score = average coverage across all ground truth combinations - duplicate penalty
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                pred_parsed = safe_parse_json(completion)
                gold_parsed = safe_parse_json(sol)

                # Check prediction format consistency issues
                penalty = self._check_consistency_penalties(pred_parsed)
                if penalty < 0:
                    rewards.append(penalty)
                    continue

                pred_is_no_comb = is_no_comb(pred_parsed)
                gold_is_no_comb = is_no_comb(gold_parsed)

                # Case 1: Both are NO_COMB - correct
                if gold_is_no_comb and pred_is_no_comb:
                    rewards.append(1.0)
                    continue

                # Case 2: Ground truth is NO_COMB but prediction has combinations - give partial score (predicted combinations may be meaningful)
                if gold_is_no_comb and not pred_is_no_comb:
                    # No ground truth combinations to compare against, give 0
                    rewards.append(0.0)
                    continue

                # Case 3: Ground truth has combinations but prediction is NO_COMB (False Negative) - severe penalty
                if not gold_is_no_comb and pred_is_no_comb:
                    rewards.append(-1.0)
                    continue

                # Case 4: Both have combinations, compute coverage
                gold_sets = extract_drug_sets(gold_parsed, min_ents=2)
                pred_sets = extract_drug_sets(pred_parsed, min_ents=2)

                if not gold_sets:
                    rewards.append(0.0)
                    continue

                if not pred_sets:
                    # Prediction has no valid drug combinations (possibly insufficient entities)
                    rewards.append(0.0)
                    continue

                # Compute maximum coverage for each ground truth combination
                coverage_scores = []
                for gold_set in gold_sets:
                    gold_drugs = set(gold_set)
                    max_coverage = 0.0

                    for pred_set in pred_sets:
                        pred_drugs = set(pred_set)
                        intersection = len(gold_drugs & pred_drugs)
                        union = len(gold_drugs | pred_drugs)

                        if union > 0 and intersection >= 2:
                            coverage = intersection / union
                            max_coverage = max(max_coverage, coverage)

                    coverage_scores.append(max_coverage)

                base_reward = sum(coverage_scores) / len(coverage_scores)

                # Compute duplicate penalty
                duplicate_penalty = calculate_duplicate_penalty(pred_parsed, min_ents=2)

                # Final reward = base coverage - duplicate penalty
                reward = max(base_reward - duplicate_penalty, -1.0)
                rewards.append(reward)

            except Exception:
                rewards.append(0.0)

        return rewards

    def _check_consistency_penalties(self, pred_parsed: List[Dict]) -> float:
        """
        Check prediction consistency issues and return penalty score.

        Returns:
            0.0: No issues
            -1.0: Consistency issue found
        """
        if not isinstance(pred_parsed, list):
            return 0.0

        for item in pred_parsed:
            if not isinstance(item, dict):
                continue

            rel_type = get_rel_type(item)
            ent_set = get_ent_set(item)
            has_entities = len(ent_set) > 0

            # Penalty 1: Predicted type is not NO_COMB but entity set is empty
            if rel_type != "NO_COMB" and rel_type != "" and not has_entities:
                return -1.0

            # Penalty 2: Predicted type is NO_COMB but entities were provided
            if rel_type == "NO_COMB" and has_entities:
                return -1.0

        return 0.0


class DrugCombAccuracyORM(ORM):
    """
    DrugComb accuracy reward: Computed based on Partial F1 and Exact F1.

    References the evaluation logic from eval_files_drugcomb.py.

    Partial Match: Two drug combinations with intersection >= 2 count as a match.
    Exact Match: Two drug combinations are identical (Jaccard >= 0.9999).

    Reward rules:
    - Considers relation type (POS, NEG, COMB)
    - Final reward = 0.33 * partial_f1 + 0.67 * exact_f1 (i.e., 1:2 weighting)
    - If the predicted JSON format is incorrect, returns 0 directly
    """

    def __init__(self):
        self.partial_weight = 0.33
        self.exact_weight = 0.67

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                # First check if prediction format is correct; return 0 directly for incorrect format
                if check_format(completion, DRUGCOMB_VALID_TYPES, min_ents=2) != 1.0:
                    # Special handling: valid NO_COMB format predictions should not be rejected here
                    pred_parsed = safe_parse_json(completion)
                    if not is_no_comb(pred_parsed):
                        rewards.append(0.0)
                        continue

                pred_parsed = safe_parse_json(completion)
                gold_parsed = safe_parse_json(sol)

                # JSON 解析失败（返回空列表且不是有效格式）直接返回 0
                if not pred_parsed and not is_no_comb(pred_parsed):
                    rewards.append(0.0)
                    continue

                pred_is_no_comb = is_no_comb(pred_parsed)
                gold_is_no_comb = is_no_comb(gold_parsed)

                # Case 1: 两者都是 NO_COMB - 完全正确
                if gold_is_no_comb and pred_is_no_comb:
                    rewards.append(1.0)
                    continue

                # Case 2: 一个是 NO_COMB，另一个不是 - 完全错误
                if gold_is_no_comb != pred_is_no_comb:
                    rewards.append(0.0)
                    continue

                # Case 3: 两者都有关系，计算 F1
                gold_rels = extract_relations(gold_parsed, min_ents=2)
                pred_rels = extract_relations(pred_parsed, min_ents=2)

                if not gold_rels:
                    rewards.append(0.0)
                    continue

                if not pred_rels:
                    rewards.append(0.0)
                    continue

                partial_f1 = self._compute_f1(gold_rels, pred_rels, exact=False)
                exact_f1 = self._compute_f1(gold_rels, pred_rels, exact=True)

                reward = self.partial_weight * partial_f1 + self.exact_weight * exact_f1
                rewards.append(reward)

            except Exception:
                rewards.append(0.0)

        return rewards

    def _compute_f1(self, gold_rels: List[Tuple], pred_rels: List[Tuple], exact: bool) -> float:
        """计算 F1 分数"""
        if not gold_rels or not pred_rels:
            return 0.0

        # Recall: 对于 gold 中的每个关系，找到 pred 中最佳匹配
        gold_scores = []
        for g_type, g_drugs in gold_rels:
            best_score = 0.0
            g_set = set(g_drugs)

            for p_type, p_drugs in pred_rels:
                if g_type != p_type:
                    continue

                p_set = set(p_drugs)
                intersection = len(g_set & p_set)
                union = len(g_set | p_set)

                if exact:
                    if union > 0 and intersection / union >= 0.9999:
                        best_score = 1.0
                        break
                else:
                    if intersection >= 2 and union > 0:
                        score = intersection / union
                        best_score = max(best_score, score)

            gold_scores.append(best_score)

        # Precision: 对于 pred 中的每个关系，找到 gold 中最佳匹配
        pred_scores = []
        for p_type, p_drugs in pred_rels:
            best_score = 0.0
            p_set = set(p_drugs)

            for g_type, g_drugs in gold_rels:
                if g_type != p_type:
                    continue

                g_set = set(g_drugs)
                intersection = len(g_set & p_set)
                union = len(g_set | p_set)

                if exact:
                    if union > 0 and intersection / union >= 0.9999:
                        best_score = 1.0
                        break
                else:
                    if intersection >= 2 and union > 0:
                        score = intersection / union
                        best_score = max(best_score, score)

            pred_scores.append(best_score)

        precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0.0
        recall = sum(gold_scores) / len(gold_scores) if gold_scores else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return f1


# ==================== DrugComb COT 奖励函数 ====================

class DrugCombCOTFormatORM(ORM):
    """
    DrugComb COT 格式奖励：检查输出是否包含正确的 COT 格式

    奖励规则：
    - 首先检查是否同时包含 <think></think> 和 <answer></answer> 标签
    - 如果不完全包含则直接不得分 (0.0)
    - 如果包含则获得 0.5 分
    - 然后检查 <answer> 标签中是否是正确的 JSON 格式
    - 如果是则再得 0.5 分，获得满分 1.0
    - 如果不是正确 JSON 格式则结束，只得 0.5 分
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            # 检查是否包含两个标签
            has_think, has_answer = has_cot_tags(completion)

            if not (has_think and has_answer):
                # 不完全包含两个标签，不得分
                rewards.append(0.0)
                continue

            # 包含两个标签，获得 0.5 分
            score = 0.5

            # 提取 answer 内容并检查 JSON 格式
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容是否为正确的 JSON 格式
            if check_format(answer_content, DRUGCOMB_VALID_TYPES, min_ents=2) == 1.0:
                score += 0.5

            rewards.append(score)

        return rewards


class DrugCombCOTThinkORM(ORM):
    """
    DrugComb COT 思考过程奖励：检查思考过程是否包含分点

    奖励规则：
    - 首先检查是否包含 <think> 标签
    - 提取 <think> 标签中的思考过程
    - 检查是否包含 [1][2][3][4] 四个分点
    - 如果包含所有四个分点则得 1.0 分
    - 如果不完全包含则得 0.0 分
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            # 检查是否包含 think 标签
            has_think, _ = has_cot_tags(completion)

            if not has_think:
                rewards.append(0.0)
                continue

            # 提取 think 内容
            think_content, _ = extract_cot_tags(completion)

            # 检查是否包含四个分点
            if check_think_points(think_content):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


class DrugCombCoverageCOTORM(ORM):
    """
    DrugComb COT 覆盖奖励：从 <answer> 标签中提取结果，然后调用覆盖率计算

    首先从 <answer> 标签中提取内容，然后使用 DrugCombCoverageORM 的逻辑计算覆盖率

    注意：如果不存在 <answer> 标签或 <answer> 内容格式错误，直接返回 0 分
    """

    def __init__(self):
        self._coverage_orm = DrugCombCoverageORM()

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            # 检查是否存在 answer 标签
            _, has_answer = has_cot_tags(completion)
            if not has_answer:
                rewards.append(0.0)
                continue

            # 提取 answer 内容
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容格式是否正确
            if not answer_content or check_format(answer_content, DRUGCOMB_VALID_TYPES, min_ents=2) != 1.0:
                rewards.append(0.0)
                continue
            # 提取solution中的answer内容
            _, gold_answer = extract_cot_tags(sol)
            # 调用原有的覆盖率计算
            coverage_reward = self._coverage_orm([answer_content], [gold_answer], **kwargs)[0]
            rewards.append(coverage_reward)

        return rewards


class DrugCombAccuracyCOTORM(ORM):
    """
    DrugComb COT 准确性奖励：从 <answer> 标签中提取结果，然后调用准确性计算

    首先从 <answer> 标签中提取内容，然后使用 DrugCombAccuracyORM 的逻辑计算准确性

    注意：如果不存在 <answer> 标签或 <answer> 内容格式错误，直接返回 0 分
    """

    def __init__(self):
        self._accuracy_orm = DrugCombAccuracyORM()

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            # 检查是否存在 answer 标签
            _, has_answer = has_cot_tags(completion)
            if not has_answer:
                rewards.append(0.0)
                continue

            # 提取 answer 内容
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容格式是否正确
            if not answer_content or check_format(answer_content, DRUGCOMB_VALID_TYPES, min_ents=2) != 1.0:
                rewards.append(0.0)
                continue

            # 提取solution中的answer内容
            _, gold_answer = extract_cot_tags(sol)
            # 调用原有的准确性计算
            accuracy_reward = self._accuracy_orm([answer_content], [gold_answer], **kwargs)[0]
            rewards.append(accuracy_reward)

        return rewards


# ==================== DDI 奖励函数 ====================

class DDIFormatORM(ORM):
    """
    DDI 格式奖励：检查输出是否为正确的 JSON 格式

    奖励规则：
    - 1.0: 输出为正确的 JSON 列表格式
    - 0.0: 格式错误
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        return [check_format(c, DDI_VALID_TYPES, min_ents=2) for c in completions]


class DDICoverageORM(ORM):
    """
    DDI 药物组合覆盖奖励：二元实体组合匹配

    DDI 是二元药物相互作用数据集，简化为检查实体组合是否正确匹配

    惩罚项：
    - False Negative: 真实有组合但预测 NO_COMB → -1.0
    - Empty Entity Set: 预测类型不是 NO_COMB 但实体集合为空 → -1.0
    - Inconsistent NO_COMB: 预测类型是 NO_COMB 但输出了实体集合 → -1.0
    - Duplicate Penalty: 预测的元组有重复时，按重复比例扣分

    覆盖率计算：
    - 对于二元关系，检查实体对是否完全匹配
    - 最终分数 = 覆盖率 - 重复惩罚
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                pred_parsed = safe_parse_json(completion)
                gold_parsed = safe_parse_json(sol)

                # 检查一致性问题
                penalty = self._check_consistency_penalties(pred_parsed)
                if penalty < 0:
                    rewards.append(penalty)
                    continue

                pred_is_no_comb = is_no_comb(pred_parsed)
                gold_is_no_comb = is_no_comb(gold_parsed)

                # Case 1: 两者都是 NO_COMB
                if gold_is_no_comb and pred_is_no_comb:
                    rewards.append(1.0)
                    continue

                # Case 2: 真实是 NO_COMB，但预测了组合
                if gold_is_no_comb and not pred_is_no_comb:
                    rewards.append(0.0)
                    continue

                # Case 3: 真实有组合，但预测 NO_COMB - 惩罚
                if not gold_is_no_comb and pred_is_no_comb:
                    rewards.append(-1.0)
                    continue

                # Case 4: 计算实体组合匹配率
                gold_sets = extract_drug_sets(gold_parsed, min_ents=2)
                pred_sets = extract_drug_sets(pred_parsed, min_ents=2)

                if not gold_sets:
                    rewards.append(0.0)
                    continue

                if not pred_sets:
                    rewards.append(0.0)
                    continue

                # DDI 是二元关系，检查精确匹配
                matched = 0
                for gold_set in gold_sets:
                    if gold_set in pred_sets:
                        matched += 1

                base_coverage = matched / len(gold_sets)

                # Compute duplicate penalty
                duplicate_penalty = calculate_duplicate_penalty(pred_parsed, min_ents=2)

                # Final reward = base coverage - duplicate penalty
                reward = max(base_coverage - duplicate_penalty, -1.0)
                rewards.append(reward)

            except Exception:
                rewards.append(0.0)

        return rewards

    def _check_consistency_penalties(self, pred_parsed: List[Dict]) -> float:
        """检查一致性问题"""
        if not isinstance(pred_parsed, list):
            return 0.0

        for item in pred_parsed:
            if not isinstance(item, dict):
                continue

            rel_type = get_rel_type(item)
            ent_set = get_ent_set(item)
            has_entities = len(ent_set) > 0

            if rel_type != "NO_COMB" and rel_type != "" and not has_entities:
                return -1.0

            if rel_type == "NO_COMB" and has_entities:
                return -1.0

        return 0.0


class DDIAccuracyORM(ORM):
    """
    DDI 准确性奖励：微平均 F1 计算

    对每个样本计算微平均 F1 作为奖励
    - 考虑关系类型（effect, advise, int, mechanism）
    - 使用精确匹配
    - 如果预测的 JSON 格式错误，直接返回 0 分
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            try:
                # 首先检查预测格式是否正确，格式错误直接返回 0
                if check_format(completion, DDI_VALID_TYPES, min_ents=2) != 1.0:
                    # 特殊处理：如果预测是有效的 NO_COMB 格式，不应该在这里被拒绝
                    pred_parsed = safe_parse_json(completion)
                    if not is_no_comb(pred_parsed):
                        rewards.append(0.0)
                        continue

                pred_parsed = safe_parse_json(completion)
                gold_parsed = safe_parse_json(sol)

                # JSON 解析失败（返回空列表且不是有效格式）直接返回 0
                if not pred_parsed and not is_no_comb(pred_parsed):
                    rewards.append(0.0)
                    continue

                pred_is_no_comb = is_no_comb(pred_parsed)
                gold_is_no_comb = is_no_comb(gold_parsed)

                # Case 1: 两者都是 NO_COMB
                if gold_is_no_comb and pred_is_no_comb:
                    rewards.append(1.0)
                    continue

                # Case 2: 一个是 NO_COMB，另一个不是
                if gold_is_no_comb != pred_is_no_comb:
                    rewards.append(0.0)
                    continue

                # Case 3: 计算微平均 F1
                gold_rels = set(extract_relations(gold_parsed, min_ents=2))
                pred_rels = set(extract_relations(pred_parsed, min_ents=2))

                if not gold_rels:
                    rewards.append(0.0)
                    continue

                if not pred_rels:
                    rewards.append(0.0)
                    continue

                # 精确匹配计算 TP, FP, FN
                tp = len(gold_rels & pred_rels)
                fp = len(pred_rels - gold_rels)
                fn = len(gold_rels - pred_rels)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0

                rewards.append(f1)

            except Exception:
                rewards.append(0.0)

        return rewards


# ==================== DDI COT 奖励函数 ====================

class DDICOTFormatORM(ORM):
    """
    DDI COT 格式奖励：检查输出是否包含正确的 COT 格式

    奖励规则：
    - 首先检查是否同时包含 <think></think> 和 <answer></answer> 标签
    - 如果不完全包含则直接不得分 (0.0)
    - 如果包含则获得 0.5 分
    - 然后检查 <answer> 标签中是否是正确的 JSON 格式
    - 如果是则再得 0.5 分，获得满分 1.0
    - 如果不是正确 JSON 格式则结束，只得 0.5 分
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            # 检查是否包含两个标签
            has_think, has_answer = has_cot_tags(completion)

            if not (has_think and has_answer):
                # 不完全包含两个标签，不得分
                rewards.append(0.0)
                continue

            # 包含两个标签，获得 0.5 分
            score = 0.5

            # 提取 answer 内容并检查 JSON 格式
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容是否为正确的 JSON 格式
            if check_format(answer_content, DDI_VALID_TYPES, min_ents=2) == 1.0:
                score += 0.5

            rewards.append(score)

        return rewards


class DDICOTThinkORM(ORM):
    """
    DDI COT 思考过程奖励：检查思考过程是否包含分点

    奖励规则：
    - 首先检查是否包含 <think> 标签
    - 提取 <think> 标签中的思考过程
    - 检查是否包含 [1][2][3][4] 四个分点
    - 如果包含所有四个分点则得 1.0 分
    - 如果不完全包含则得 0.0 分
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion in completions:
            # 检查是否包含 think 标签
            has_think, _ = has_cot_tags(completion)

            if not has_think:
                rewards.append(0.0)
                continue

            # 提取 think 内容
            think_content, _ = extract_cot_tags(completion)

            # 检查是否包含四个分点
            if check_think_points(think_content):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


class DDICoverageCOTORM(ORM):
    """
    DDI COT 覆盖奖励：从 <answer> 标签中提取结果，然后调用覆盖率计算

    首先从 <answer> 标签中提取内容，然后使用 DDICoverageORM 的逻辑计算覆盖率

    注意：如果不存在 <answer> 标签或 <answer> 内容格式错误，直接返回 0 分
    """

    def __init__(self):
        self._coverage_orm = DDICoverageORM()

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            # 检查是否存在 answer 标签
            _, has_answer = has_cot_tags(completion)
            if not has_answer:
                rewards.append(0.0)
                continue

            # 提取 answer 内容
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容格式是否正确
            if not answer_content or check_format(answer_content, DDI_VALID_TYPES, min_ents=2) != 1.0:
                rewards.append(0.0)
                continue

            # 提取 solution 中的 answer 内容
            _, gold_answer = extract_cot_tags(sol)

            # 调用原有的覆盖率计算
            coverage_reward = self._coverage_orm([answer_content], [gold_answer], **kwargs)[0]
            rewards.append(coverage_reward)

        return rewards


class DDIAccuracyCOTORM(ORM):
    """
    DDI COT 准确性奖励：从 <answer> 标签中提取结果，然后调用准确性计算

    首先从 <answer> 标签中提取内容，然后使用 DDIAccuracyORM 的逻辑计算准确性

    注意：如果不存在 <answer> 标签或 <answer> 内容格式错误，直接返回 0 分
    """

    def __init__(self):
        self._accuracy_orm = DDIAccuracyORM()

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for completion, sol in zip(completions, solution):
            # 检查是否存在 answer 标签
            _, has_answer = has_cot_tags(completion)
            if not has_answer:
                rewards.append(0.0)
                continue

            # 提取 answer 内容
            _, answer_content = extract_cot_tags(completion)

            # 检查 answer 内容格式是否正确
            if not answer_content or check_format(answer_content, DDI_VALID_TYPES, min_ents=2) != 1.0:
                rewards.append(0.0)
                continue

            # 提取 solution 中的 answer 内容
            _, gold_answer = extract_cot_tags(sol)

            # 调用原有的准确性计算
            accuracy_reward = self._accuracy_orm([answer_content], [gold_answer], **kwargs)[0]
            rewards.append(accuracy_reward)

        return rewards


# ==================== 注册奖励函数 ====================

# DrugComb 奖励函数
orms['drugcomb_format'] = DrugCombFormatORM
orms['drugcomb_coverage'] = DrugCombCoverageORM
orms['drugcomb_accuracy'] = DrugCombAccuracyORM

# DrugComb COT 奖励函数
orms['drugcomb_cot_format'] = DrugCombCOTFormatORM
orms['drugcomb_cot_think'] = DrugCombCOTThinkORM
orms['drugcomb_coverage_cot'] = DrugCombCoverageCOTORM
orms['drugcomb_accuracy_cot'] = DrugCombAccuracyCOTORM

# DDI 奖励函数
orms['ddi_format'] = DDIFormatORM
orms['ddi_coverage'] = DDICoverageORM
orms['ddi_accuracy'] = DDIAccuracyORM

# DDI COT 奖励函数
orms['ddi_cot_format'] = DDICOTFormatORM
orms['ddi_cot_think'] = DDICOTThinkORM
orms['ddi_coverage_cot'] = DDICoverageCOTORM
orms['ddi_accuracy_cot'] = DDIAccuracyCOTORM


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("DrugComb 奖励函数测试")
    print("=" * 80)

    drugcomb_test_cases = [
        # (prediction, gold, description)
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完全正确"
        ),
        (
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "NO_COMB 正确"
        ),
        (
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "False Negative: 真实有组合但预测 NO_COMB (应该得-1)"
        ),
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "真实 NO_COMB 但预测了组合"
        ),
        (
            "[{'type': 'POS', 'ent_set': []}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "预测类型非 NO_COMB 但实体为空 (应该得-1)"
        ),
        (
            "[{'type': 'NO_COMB', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "预测 NO_COMB 但输出了实体 (应该得-1)"
        ),
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2', 'drug3']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "部分匹配"
        ),
        (
            "[{'type': 'NEG', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "类型错误"
        ),
        (
            "invalid json",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "格式错误"
        ),
    ]

    format_orm = DrugCombFormatORM()
    coverage_orm = DrugCombCoverageORM()
    accuracy_orm = DrugCombAccuracyORM()

    for pred, gold, desc in drugcomb_test_cases:
        format_reward = format_orm([pred], [gold])[0]
        coverage_reward = coverage_orm([pred], [gold])[0]
        accuracy_reward = accuracy_orm([pred], [gold])[0]

        print(f"\n测试: {desc}")
        print(f"  预测: {pred[:60]}...")
        print(f"  真实: {gold[:60]}...")
        print(f"  格式: {format_reward:.2f} | 覆盖: {coverage_reward:.2f} | 准确: {accuracy_reward:.2f}")

    print("\n" + "=" * 80)
    print("DDI 奖励函数测试")
    print("=" * 80)

    ddi_test_cases = [
        (
            "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]",
            "完全正确"
        ),
        (
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "NO_COMB 正确"
        ),
        (
            "[{'type': 'NO_COMB', 'ent_set': []}]",
            "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]",
            "False Negative (应该得-1)"
        ),
        (
            "[{'type': 'mechanism', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]",
            "类型错误"
        ),
    ]

    ddi_format_orm = DDIFormatORM()
    ddi_coverage_orm = DDICoverageORM()
    ddi_accuracy_orm = DDIAccuracyORM()

    for pred, gold, desc in ddi_test_cases:
        format_reward = ddi_format_orm([pred], [gold])[0]
        coverage_reward = ddi_coverage_orm([pred], [gold])[0]
        accuracy_reward = ddi_accuracy_orm([pred], [gold])[0]

        print(f"\n测试: {desc}")
        print(f"  预测: {pred[:60]}...")
        print(f"  真实: {gold[:60]}...")
        print(f"  格式: {format_reward:.2f} | 覆盖: {coverage_reward:.2f} | 准确: {accuracy_reward:.2f}")

    print("\n" + "=" * 80)
    print("DrugComb COT 奖励函数测试")
    print("=" * 80)

    cot_test_cases = [
        # (prediction, gold, description)
        (
            "<think>[1] 分析药物... [2] 检查相互作用... [3] 评估效果... [4] 得出结论...</think><answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完整 COT 格式 [1][2][3][4]，正确答案"
        ),
        (
            "<think>1. 分析药物 2. 检查相互作用 3. 评估效果 4. 得出结论</think><answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完整 COT 格式 1. 2. 3. 4.，正确答案"
        ),
        (
            "<think>首先分析药物，其次检查相互作用，然后评估效果，最后得出结论</think><answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完整 COT 格式 首先/其次/然后/最后，正确答案"
        ),
        (
            "<think>第一步分析，第二步检查，第三步评估，第四步结论</think><answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完整 COT 格式 第一/第二/第三/第四，正确答案"
        ),
        (
            "<think>思考过程没有分点</think><answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "COT 格式正确但思考没有分点"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think><answer>invalid json</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "思考有分点但答案格式错误 (COT覆盖和准确应为0)"
        ),
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "没有 COT 标签 (COT覆盖和准确应为0)"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "只有 think 标签没有 answer (COT覆盖和准确应为0)"
        ),
        (
            "<answer>[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "只有 answer 标签没有 think (COT格式0, 但COT覆盖和准确可计算)"
        ),
    ]

    cot_format_orm = DrugCombCOTFormatORM()
    cot_think_orm = DrugCombCOTThinkORM()
    cot_coverage_orm = DrugCombCoverageCOTORM()
    cot_accuracy_orm = DrugCombAccuracyCOTORM()

    for pred, gold, desc in cot_test_cases:
        format_reward = cot_format_orm([pred], [gold])[0]
        think_reward = cot_think_orm([pred], [gold])[0]
        coverage_reward = cot_coverage_orm([pred], [gold])[0]
        accuracy_reward = cot_accuracy_orm([pred], [gold])[0]

        print(f"\n测试: {desc}")
        print(f"  预测: {pred[:80]}...")
        print(f"  真实: {gold[:60]}...")
        print(f"  COT格式: {format_reward:.2f} | 思考分点: {think_reward:.2f} | COT覆盖: {coverage_reward:.2f} | COT准确: {accuracy_reward:.2f}")

    print("\n" + "=" * 80)
    print("重复惩罚测试")
    print("=" * 80)

    duplicate_test_cases = [
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}, {'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "完全重复的预测 (应该有惩罚)"
        ),
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}, {'type': 'POS', 'ent_set': ['drug3', 'drug4']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "不重复的预测 (无惩罚)"
        ),
        (
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}, {'type': 'POS', 'ent_set': ['drug1', 'drug2']}, {'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "[{'type': 'POS', 'ent_set': ['drug1', 'drug2']}]",
            "三次重复的预测 (应该有更大惩罚)"
        ),
    ]

    for pred, gold, desc in duplicate_test_cases:
        coverage_reward = coverage_orm([pred], [gold])[0]
        parsed = safe_parse_json(pred)
        penalty = calculate_duplicate_penalty(parsed)

        print(f"\n测试: {desc}")
        print(f"  重复惩罚: {penalty:.2f} | 最终覆盖奖励: {coverage_reward:.2f}")

    print("\n" + "=" * 80)
    print("DDI COT 奖励函数测试")
    print("=" * 80)

    ddi_cot_test_cases = [
        # (prediction, gold, description)
        (
            "<think>[1] 分析药物... [2] 检查相互作用... [3] 评估效果... [4] 得出结论...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "完整 COT 格式，正确答案"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think><answer>[{'type': 'NO_COMB', 'ent_set': []}]</answer>",
            "<think>...</think><answer>[{'type': 'NO_COMB', 'ent_set': []}]</answer>",
            "NO_COMB 正确"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think><answer>[{'type': 'NO_COMB', 'ent_set': []}]</answer>",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "False Negative (覆盖应为-1)"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think><answer>[{'type': 'mechanism', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "类型错误 (覆盖正确，准确为0)"
        ),
        (
            "<think>思考过程没有分点</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "COT 格式正确但思考没有分点"
        ),
        (
            "<think>[1] 分析 [2] 检查 [3] 评估 [4] 结论</think><answer>invalid json</answer>",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "思考有分点但答案格式错误"
        ),
        (
            "[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]",
            "<think>...</think><answer>[{'type': 'effect', 'ent_set': ['drug1', 'drug2']}]</answer>",
            "没有 COT 标签"
        ),
    ]

    ddi_cot_format_orm = DDICOTFormatORM()
    ddi_cot_think_orm = DDICOTThinkORM()
    ddi_cot_coverage_orm = DDICoverageCOTORM()
    ddi_cot_accuracy_orm = DDIAccuracyCOTORM()

    for pred, gold, desc in ddi_cot_test_cases:
        format_reward = ddi_cot_format_orm([pred], [gold])[0]
        think_reward = ddi_cot_think_orm([pred], [gold])[0]
        coverage_reward = ddi_cot_coverage_orm([pred], [gold])[0]
        accuracy_reward = ddi_cot_accuracy_orm([pred], [gold])[0]

        print(f"\n测试: {desc}")
        print(f"  预测: {pred[:80]}...")
        print(f"  真实: {gold[:60]}...")
        print(f"  COT格式: {format_reward:.2f} | 思考分点: {think_reward:.2f} | COT覆盖: {coverage_reward:.2f} | COT准确: {accuracy_reward:.2f}")

    print("\n" + "=" * 80)
