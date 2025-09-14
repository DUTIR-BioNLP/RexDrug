import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
import time

Analyst = OpenAI(
      api_key=""
  )
Reviewer = OpenAI(
      api_key=""
  )
# JSON 解析函数
def safe_extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                print("JSON parse error:", e)
        return None

# 生成推理思维链的 prompt
new_prompt = """
You are a biomedical expert tasked with generating a high-quality reasoning chain to extract drug combinations and their effects from the Target sentence. You can refer to the Reference Context to assist with reasoning. Please generate the process of reasoning out this answer based on the given correct answer. The drug combination relationships include the following three types: \n'POS': The drug combination has a positive effect. \n'NEG': The drug combination has a negative effect. \n'COMB': The drug combination has no significant positive or negative effect.

Your reasoning must be:
- Medically accurate
- Naturally written, as if independently derived by a human expert
- Focused on identifying and analyzing **drug combination relationships**, especially higher-order combinations (involving three or more drugs)

Important Constraints:
- This is the most important! If your initial reasoning diverges from the correct answer, self-correct through logic only, and align with the gold-standard RE results, without revealing that you knew the answer. 
- This is the most important part: reason step by step and stop after arriving at the answer. Do not compare it with the Ground Truth Answer. Avoid excessive summaries.
- Keep the reasoning concise! The thinking process should be no more than 300 words, and ideally around 250 words.
- Please provide a well-structured response that reasons point by point, with clear logical progression.
"""


# 评分 prompt
evaluation_prompt = """You are an expert with strong knowledge in the biomedical domain. Your task is to evaluate the quality of a reasoning chain used in a drug combination extraction task.\nYou will be provided with the following four components:Target Sentence, Reference Context, Ground Truth Answer, Reasoning Chain.\nPlease assess the quality of the reasoning chain based on the following six criteria. Each criterion should be scored on a scale from 1 to 5 (1 = very poor, 5 = excellent):
1. **Medical Soundness**: Does the reasoning contain medically accurate information, without factual errors or violations of biomedical knowledge?
2. **Semantic Alignment**: Does the reasoning accurately reflect the meaning of the target sentence and the reference context, without deviating from the original intent?
3. **Combination Coverage**: Does the reasoning sufficiently consider complex drug combinations, especially higher-order relations (three or more drugs involved)?
4. **Reasoning Accuracy**: This is more important! Does the reasoning chain ultimately lead to the correct drug combination and effect as provided in the ground truth answer? If RE results is wrong, assign a score of 0.
5. **Naturalness of Reasoning Origin**: This is the most Important! If the reasoning process involves comparisons with or references to the **ground truth**, it indicates that the model may have had prior knowledge of the correct answer, assign a score of 0.
6. **Reasoning Length**: Is the reasoning chain appropriately concise? If the total word count exceeds 400, assign a score of 0.
Please pay special attention to score_naturalness. Please output your evaluation in the following JSON format:
{"score_medical": int, "score_semantic": int, "score_coverage": int, "score_accuracy": int, "score_naturalness": int, "score_length": int,"comment": "A short summary comment"}\n
"""

# 保存函数
def save_jsonl(path, data):
    with open(path, 'a+') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 思维链生成：替换为 GPT-4o
def call_openai_reasoner(target_sentence, answer, context, failed_examples=None):
    if failed_examples:
        examples_str = "\n\n".join(
            [f"Failed Example {i+1}:\nReasoning Chain: {ex['think']}\nReview Comment: {ex['comment']}" for i, ex in enumerate(failed_examples)]
        )
        prompt = f"""Below are previous failed examples and review comments. Avoid the same issues:\n{examples_str}\n\nThe Target Sentence: {target_sentence}\nReference Context: {context}\nGround Truth Answer: {answer}\n{new_prompt}"""
    else:
        prompt = f"""The Target Sentence: {target_sentence}\nReference Context: {context}\nGround Truth Answer: {answer}\n\n{new_prompt}"""

    response = Analyst.chat.completions.create(
        model="gpt-4o",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.6
    )
    output = response.choices[0].message.content
    return output

# GPT-4o 打分评估
def call_openai_evaluator(target_sentence, context, answer, think_chain):
    eval_input = f"""{evaluation_prompt}\n\nTarget Sentence: {target_sentence}\nReference Context: {context}\nGround Truth Answer: {answer}\nReasoning Chain: {think_chain}"""
    response = Reviewer.chat.completions.create(
        model="gpt-4o",
        messages=[{'role': 'user', 'content': eval_input}],
        temperature=0.0
    )
    raw_output = response.choices[0].message.content
    try:
        eval_json = safe_extract_json(raw_output)
    except:
        eval_json = {
            "score_medical": 0,
            "score_semantic": 0,
            "score_coverage": 0,
            "score_accuracy": 0,
            "score_naturalness":0,
            "score_length":0,
            "comment": "Model parsing failed"
        }
    return eval_json

# 主流程
with open("train.json", 'r') as f:
    Data = json.load(f)

MAX_RETRIES = 2
MIN_ACCEPTABLE_SCORE = 4
ids = 0

for row in tqdm(Data, total=len(Data), desc="Processing"):
    ids += 1
    problem = row['problem']
    answer = row['solution']
    sentence = row['target_sentence']
    paragraph = row['paragraph']
    retries = 0
    failed_attempts = []
    log_entry = {"ids": ids, "attempts": []}

    while retries < MAX_RETRIES:
        try:
            output_text = call_openai_reasoner(sentence, answer, failed_attempts)
            score = call_openai_evaluator(sentence, paragraph, answer, output_text)
            time.sleep(0.2)  # 确保 API 调用间隔

            # 日志保存
            attempt_log = {
                "retry": retries,
                "think": output_text,
                "score": score
            }
            log_entry["attempts"].append(attempt_log)
            save_jsonl("log.jsonl", [log_entry])

            # 保存首次推理
            if retries == 0:
                first_outputs=[{
                    "ids": ids,
                    "input": problem,
                    "true_answer": answer,
                    "think": output_text
                }]
                save_jsonl("the_first_reasoning.jsonl", first_outputs)

            # 检查是否通过评分门槛
            if all(score[k] >= MIN_ACCEPTABLE_SCORE for k in ['score_medical', 'score_semantic', 'score_coverage', 'score_accuracy', 'score_naturalness', 'score_length']):
                final_outputs=[{
                    "ids": ids,
                    "input": problem,
                    "true_answer": answer,
                    "think": output_text,
                    "retry_num": retries
                }]
                save_jsonl("the_last_reasoning.jsonl", final_outputs)
                break
            else:
                failed_attempts.append({
                    "think": output_text,
                    "comment": score["comment"]
                })

            retries += 1

        except Exception as e:
            print(f"Error in processing item {ids}: {e}")
            break
    else:
        fail_data = [{
            "ids": ids,
            "input": problem,
            "true_answer": answer,
            "think": output_text,
            "retry_num": retries
        }]
        save_jsonl("try_fail_data.json", fail_data)
