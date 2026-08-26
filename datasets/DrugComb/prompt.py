"""Prompts used for DrugComb reasoning distillation and model training.

The main dual-agent pipeline uses ``RE_COT_TEMPLATE`` for the medical reasoning
analyst and ``evaluation_prompt`` for the expert reviewer. ``RE_TEMPLATE`` is
the label-blind analyst prompt used only for the rejection-sampling control.
The accepted traces are converted to SFT/GRPO chat records with
``SYSTEM_PROMPT`` and ``USER_PROMPT_TEMPLATE``.
"""


evaluation_prompt = """
You are a biomedical domain expert and an impartial evaluator.

Your task is to evaluate the quality of the reasoning chain and relation extraction results
for a drug combination extraction case.

You will receive the following content:
- A Target Sentence
- A Reference Context
- Expert-validated drug combination relations (GOLD_RE)
- Model Reasoning (inside <think>...</think>)
- Model Answer (in JSON format inside <answer>...</answer>)

Evaluate the reasoning and answer using the following six criteria.
Score each criterion from 0 to 5:
5: No substantive issues are present. The reasoning and answer are accurate, complete, and well supported, and the progression from textual evidence to the final conclusion is natural, rigorous, and consistent.
4: The output is correct and reliable overall and meets the standard for acceptance and direct use. It contains only minor localized issues that do not affect the key evidence chain, relation judgment, or final conclusion; if there is clear insufficient evidence, a logical gap, an internal contradiction, or an unreasonable inference, a score of 4 must not be assigned.
3: The output is basically usable but contains clear and non-negligible problems, such as insufficient evidence, missing important information, reasoning jumps, or logical gaps. Substantive revision is required before it can be used as high-quality data.
2: The output contains important problems that affect the main analysis, and parts of the reasoning are unreasonable or unreliable. Although some content remains useful, the key evidence chain, relation judgment, or medical interpretation has clear deficiencies.
1: The output contains serious errors, and most of the reasoning lacks textual support, is unreasonable, or contradicts the text. Its overall reliability is very low, and it is generally not usable without major correction.
0: The output completely fails to meet the task requirements, contains fundamental contradictions or severe errors, or violates a hard constraint. It has no usable value.

1. score_format
- Check whether the output follows the required structure:
  - The reasoning is enclosed in <think>...</think>.
  - The reasoning contains four sections labeled [1], [2], [3], and [4]. Each section heading is on its own line.
  - The content under each section is mainly written as bullet points beginning with "- ".
  - The answer is enclosed in <answer>...</answer>.
  - The content inside <answer> is a valid JSON array that can be parsed by json.loads.
- Assign a score of 0 if the JSON is invalid or the overall structure seriously violates the requirements.

2. score_extraction
- Normalize leading and trailing whitespace, letter case, entity order, and the order of multiple relations. Then compare the Model Answer JSON with GOLD_RE.
- Assign a score of 5 if they match exactly after normalization.
- Assign a score of 0 for any mismatch in relation type or combination boundaries, or for any missing or extra result.
- Do not use scores from 1 to 4 for this criterion.

3. score_semantic
- Are the drug combinations, treatment or experimental effects, and supporting rationale in the reasoning supported by the Target Sentence or Reference Context?
- Does the reasoning distinguish evidence from the Target Sentence from supporting information in the Reference Context?
- Does the reasoning invent any treatment regimen, patient characteristic, mechanism of action, or clinical outcome?
- Is the reasoning consistent with the final conclusion?
- If the reasoning contradicts the conclusion, or includes unsupported content that affects the relation decision, the score must not exceed 2.

4. score_medical
- Are the explanations of the disease, drug use, and treatment or experimental effects medically sound?
- Does the reasoning incorrectly interpret efficacy, toxicity, synergy, antagonism, or a combination treatment relationship?
- Do not require additional external medical knowledge. Focus on whether the existing explanation contains medical errors.
- Assign a score of 0 or 1 if there is a serious medical error.

5. score_naturalness
- Does the reasoning read like an analysis formed naturally after reading the text?
- Does it directly or indirectly reveal prior knowledge of the human-provided answer?
- Does it use meta-level phrases such as "gold answer", "given relation", "result to match", or "according to the label"?
- Is the reasoning concise, coherent, and free of repetition?
- Assign a score of 0 if the reasoning clearly reveals the internal answer anchor.

6. score_reasoning
- Does the reasoning identify the relevant candidate drugs and determine from the text whether they truly form a combination?
- Does it correctly determine the entity boundaries of binary or higher-order combinations?
- Does it reasonably distinguish POS, NEG, COMB, and NO_COMB based on the treatment or experimental effects?
- Is the reasoning concise and clear? Does it help clinicians understand why the drugs in this case should or should not be extracted as a combination?

Length consideration:
- The reasoning inside <think> should be concise. It should usually contain about 100 to 200 words and must not exceed 350 words.
- If it is too long, or too short to contain useful information, reduce score_reasoning.

Output:
Return the evaluation as a single JSON object with the following fields:

{
  "score_format": int,
  "score_extraction": int,
  "score_semantic": int,
  "score_medical": int,
  "score_naturalness": int,
  "score_reasoning": int,
  "comment": "A brief overall comment in 1 to 3 sentences."
}

Before assigning any scores:
- Carefully analyze the Target Sentence, the Reference Context, and the requirements of the relation extraction task.
- Except for score_extraction, which requires direct comparison with GOLD_RE, use GOLD_RE only as a reference for assessing the final relation result.
- For all other criteria, prioritize the quality and correctness of the reasoning process itself, including its evidential support, logical coherence, and natural progression from the textual evidence to the conclusion.
- Even when the final answer exactly matches GOLD_RE, deduct points rigorously if the reasoning is implausible, unsupported by the text, internally inconsistent, driven backward from the relation label, or unnatural in how it reaches the answer. Never award a high score merely because the final answer matches GOLD_RE.
- Score the output objectively and rigorously! Do not award high scores too readily!
"""



# Reasoning chain generation prompt. This complete template uses placeholders and follows the required output format.
RE_COT_TEMPLATE = """You are a biomedical clinician-researcher and an expert in drug combination relation extraction.

You will receive the following content:
- A Target Sentence.
- A Reference Context.
- A manually annotated and expert-validated drug combination extraction result (GOLD_RE).

Your task:
Write a short, structured, and medically interpretable reasoning chain that:
(1) explains which drug entities are relevant to the task in this case;
(2) explains which drug combinations should be extracted and determines their effects;
and remains consistent with the expert-validated extraction result.

Important requirements:
- Write all reasoning and answer content in English only.
- Treat the expert-validated extraction result (GOLD_RE) as an internal answer anchor.
- Use it only to constrain the entity scope, relation types, and completeness of the final extraction. This prevents the reasoning from diverging from the human annotation.
- Every claim in the reasoning must come from the Target Sentence or Reference Context. Do not invent medical facts that are not provided in the text.
- Do not use GOLD_RE itself as evidence.
- Do not fabricate an explanation by reasoning backward from a relation label. Do not add medical facts that are absent from the Target Sentence and Reference Context.
- If the textual evidence is weak or ambiguous, use cautious language that remains faithful to the text. Do not overstate the strength of the evidence.
- The final output must appear to be derived naturally from the Target Sentence and Reference Context alone.
- Readers should not be able to tell from the reasoning that the model received GOLD_RE in advance.
- Do not directly or indirectly mention the human-provided answer or any internal constraint.
- Prohibited phrases include, but are not limited to, "gold answer", "ground truth", "human annotation", "known answer", "result to match", "according to the label", and "the correct result shows".
- You must follow the labeling guidelines below.


---------------- LABELING GUIDELINES ----------------
There are four relation types: POS, NEG, COMB, and NO_COMB.

- POS:
  Assign POS when the text clearly indicates that a set of drugs is used in combination or forms a specific treatment regimen. The combination must be associated with a beneficial treatment or experimental effect in the current context. Examples include an improved treatment response, longer survival, or clear synergistic activity.

- NEG:
  Assign NEG when the text clearly states that the combination increases toxicity, reduces efficacy, or should not be used together. Examples include increased toxicity or risk, reduced efficacy, or an explicit recommendation to avoid the combination.

- COMB:
  Assign COMB when the text clearly states that the drugs are used together. Also assign COMB when they are regarded as a combination regimen or interaction. Examples include drugs in the same chemotherapy regimen, coadministered drugs, or drugs used in the same treatment course or experimental setting. The overall treatment or experimental effect must be neutral or cannot be clearly classified as beneficial or harmful.

- NO_COMB:
  Assign NO_COMB when the text mentions multiple drugs but only lists or compares them. Also assign NO_COMB when the drugs occur in separate treatment regimens or background information. There must be no clear evidence that they form a single combination regimen or interaction set in the current context.
  Typical NO_COMB patterns include alternative treatments (A or B) and different trial arms (A versus B). They also include previous regimens or other regimens not currently being evaluated. Another pattern is a purely theoretical or unconfirmed interaction that lacks clear evidence of actual coadministration and its outcome.
----------------------------------------------------

---------------- INPUT ----------------
Target Sentence:
{TARGET_SENTENCE}

Reference Context:
{REFERENCE_CONTEXT}

Expert-validated drug combinations and the relation type of each combination (GOLD_RE, this is the correct set that you must match, in valid JSON format):
{GOLD_RE}
---------------------------------------

Follow the exact format below when answering:

1. First, output your reasoning inside <think>...</think>.

   Inside <think>, write the following four numbered sections using bullet points that begin with "- ":

   [1] Clinical scenario
   - Write 1 to 2 bullet points.
    - Briefly describe the disease, treatment, or experimental setting discussed in the text.
    - State whether the Target Sentence concerns an actual combination treatment, experimental co-treatment, regimen comparison, safety issue, or general background information.

   [2] Candidate drugs and regimen focus
   - Write 2 to 4 bullet points.
   - Identify the drugs in the Target Sentence and relevant context that are directly relevant to the current treatment, experiment, or interaction decision.
   - Focus on drugs that the text explicitly describes as used together, compared, or interacting. Explain why they are the core drugs.
   - Mention other drugs only when they have a supportive, alternative, or background role. Clearly state that role.

   [3] Combination reasoning and clinical effect
   - Write 3 to 6 bullet points. You may add more when the case contains multiple relations.
   - Determine whether the candidate drugs are coadministered or co-treated within the same treatment regimen, trial arm, or experimental condition.
   - Cite the relevant textual evidence.
   - If a combination exists, identify the exact drugs in the combination. Preserve the complete combination when it contains three or more drugs.
   - If no combination exists, briefly explain why. For example, the drugs may only be listed or may be alternatives. They may belong to different trial arms or represent previous treatment or background information.
     - For POS, explain why the combination is beneficial.
     - For NEG, explain why the combination is harmful or reduces efficacy.
     - For COMB, explain that the drugs are clearly used together, but the overall effect is neutral or cannot be clearly determined.
     - For NO_COMB, explain why the drugs do not form a single combination regimen or interaction in the current context, even though multiple drugs are mentioned.

   [4] Extraction-oriented clinical summary
   - Write 2 to 4 bullet points.
   - In clinician-friendly language, summarize which drug combinations should be extracted under the guidelines and describe their treatment or experimental effects.
   - State the conclusion explicitly. For example, "There is no extractable combination in this sentence, so the relation can be classified as NO_COMB." Another example is, "A and B form the key harmful interaction, so the relation type can be classified as NEG."
   - Present the conclusion as a brief rationale that another clinician reading the case can understand.

   Requirements for the content inside <think>:
   - Each section heading ([1] through [4]) must be on its own line. For example, one line should contain only "[1] Clinical scenario".
   - Under each heading, use only bullet points beginning with "- ". Do not write free-text paragraphs.
   - Keep each bullet point short, usually one sentence. Make each point clinically oriented and specific to the current case.
   - Keep the overall reasoning concise. It should usually contain 100 to 200 words and must not exceed 350 words.

2. Immediately after </think>, output the final relation extraction result. Place it only inside <answer> tags.
   - Inside <answer>, output a valid JSON array with no additional text.
   - The JSON must follow this structure:

<answer>
[{{"type": "POS|NEG|COMB|NO_COMB", "ent_set": ["drug1", "drug2", "..."]}},...]
</answer>

Constraints:
- The JSON inside <answer> must be syntactically valid and directly parseable by Python json.loads.
- The output must exactly match the expert-validated extraction result in the input.
- The relation types, drugs, and spelling must be identical. You may use the same order as in {GOLD_RE}."""


# Gold-label-free reasoning template for rejection-sampling distillation.
RE_TEMPLATE = """You are a biomedical clinician-researcher and an expert in drug combination relation extraction.

You will receive the following content:
- A Target Sentence.
- A Reference Context.

Your task:
Write a short, structured, and medically interpretable reasoning chain that:
(1) extracts the drug combinations mentioned in the Target Sentence, using the Reference Context only for disambiguation and effect classification;
(2) explains which drug entities are relevant to the task in this case;
(3) explains which drug combinations should be extracted and determines their effects.

Important requirements:
- Write all reasoning and answer content in English only.
- Every claim in the reasoning must be supported by the Target Sentence or Reference Context. Do not invent medical facts that are not provided in the text.
- Extract only drug combinations that are explicitly mentioned or clearly referred to in the Target Sentence.
- Use the Reference Context only to disambiguate the Target Sentence or to provide supporting evidence about the joint use or effects of a combination identified in the Target Sentence.
- Do not extract combinations that appear only in the Reference Context and are neither mentioned nor referred to in the Target Sentence.
- You must follow the labeling guidelines below.

---------------- LABELING GUIDELINES ----------------
There are four relation types: POS, NEG, COMB, and NO_COMB.

- POS:
  Assign POS when the text clearly indicates that a set of drugs is used in combination or forms a specific treatment regimen, and the combination is associated with a beneficial treatment or experimental effect in the current context. Examples include an improved treatment response, longer survival, or clear synergistic activity.

- NEG:
  Assign NEG when the text clearly states that the combination increases toxicity, reduces efficacy, or should not be used together. Examples include increased toxicity or risk, reduced efficacy, or an explicit recommendation to avoid the combination.

- COMB:
  Assign COMB when the text clearly states that the drugs are used together or are regarded as a combination regimen or interaction. Examples include drugs in the same chemotherapy regimen, coadministered drugs, or drugs used in the same treatment course or experimental setting. The overall treatment or experimental effect must be neutral or cannot be clearly classified as beneficial or harmful.

- NO_COMB:
  Assign NO_COMB when the text mentions multiple drugs but only lists or compares them, or when the drugs occur in separate treatment regimens or background information. There must be no clear evidence that they form a single combination regimen or interaction set in the current context.
  Typical NO_COMB patterns include alternative treatments (A or B), different trial arms (A versus B), previous regimens, or other regimens not currently being evaluated. They also include purely theoretical or unconfirmed interactions that lack clear evidence of actual combined use and its outcome.
----------------------------------------------------

---------------- INPUT ----------------
Target Sentence:
{TARGET_SENTENCE}

Reference Context:
{REFERENCE_CONTEXT}

---------------------------------------

Follow the exact format below when answering:

1. First, output your reasoning inside <think>...</think>.

   Inside <think>, write the following four numbered sections using bullet points that begin with "- ":

   [1] Clinical scenario
   - Write 1 to 2 bullet points.
    - Briefly describe the disease, treatment, or experimental setting discussed in the text.
    - State whether the Target Sentence concerns an actual combination treatment, experimental co-treatment, regimen comparison, safety issue, or general background information.

   [2] Candidate drugs and regimen focus
   - Write 2 to 4 bullet points.
   - Identify the drugs in the Target Sentence and relevant context that are directly relevant to the current treatment, experiment, or interaction decision.
   - Focus on drugs that the text explicitly describes as used together, compared, or interacting. Explain why they are the core drugs.
   - Mention other drugs only when they have a supportive, alternative, or background role. Clearly state that role.

   [3] Combination reasoning and clinical effect
   - Write 3 to 6 bullet points. You may add more when the case contains multiple relations.
   - Determine whether the candidate drugs are coadministered or co-treated within the same treatment regimen, trial arm, or experimental condition, and cite the relevant textual evidence.
   - If a combination exists, identify the exact drugs in the combination. Preserve the complete combination when it contains three or more drugs.
   - If no combination exists, briefly explain why. For example, the drugs may only be listed or may be alternatives, belong to different trial arms, represent previous treatment, or appear only as background information.
     - For POS, explain why the combination is beneficial.
     - For NEG, explain why the combination is harmful or reduces efficacy.
     - For COMB, explain that the drugs are clearly used together, but the overall effect is neutral or cannot be clearly determined.
     - For NO_COMB, explain why the drugs do not form a single combination regimen or interaction in the current context, even though multiple drugs are mentioned.

   [4] Extraction-oriented clinical summary
   - Write 2 to 4 bullet points.
   - In clinician-friendly language, summarize which drug combinations should be extracted under the guidelines and describe their treatment or experimental effects.
   - State the conclusion explicitly. For example, "There is no extractable combination in this sentence, so the relation can be classified as NO_COMB." Another example is, "A and B form the key harmful interaction, so the relation type can be classified as NEG."
   - Present the conclusion as a brief rationale that another clinician reading the case can understand.

   Requirements for the content inside <think>:
   - Each section heading ([1] through [4]) must be on its own line. For example, one line should contain only "[1] Clinical scenario".
   - Under each heading, use only bullet points beginning with "- ". Do not write free-text paragraphs.
   - Keep each bullet point short, usually one sentence. Make each point clinically oriented and specific to the current case.
   - Keep the overall reasoning concise. It should usually contain 100 to 200 words and must not exceed 350 words.

2. Immediately after </think>, output the final relation extraction result. Place it only inside <answer> tags.
   - Inside <answer>, output a valid JSON array with no additional text.
   - The JSON must follow this structure:

<answer>
[{{"type": "POS|NEG|COMB|NO_COMB", "ent_set": ["drug1", "drug2", "..."]}},...]
</answer>

Constraints:
- The JSON inside <answer> must be syntactically valid and directly parseable by Python json.loads.
- When a complete regimen contains three or more drugs, output it as one higher-order combination rather than decomposing it into multiple binary combinations.
- Output multiple independent combinations as separate relations and remove duplicate relations.
- If no combination exists, the output must be exactly:
  [{{"type":"NO_COMB","ent_set":[]}}]
- If any POS, NEG, or COMB relation exists, do not include NO_COMB in the same output.
- The relation types and entity sets stated in the reasoning conclusion must exactly match those in <answer>."""


# Prompts used in the final SFT/GRPO chat records.
SYSTEM_PROMPT = (
    "You are an expert in biomedical drug-drug relation extraction.\n\n"
    "Your goal is to decide whether a sentence (with its context) contains any clinically relevant "
    "drug combination relations, and to explain your reasoning.\n\n"
    "Use these relation types:\n"
    "- POS: the drugs are used together or as a regimen and the text clearly describes a beneficial effect "
    "(e.g. better response, survival, or clear synergy).\n"
    "- NEG: the drugs are used together or interact and the text clearly describes a harmful effect "
    "(e.g. more toxicity or risk, reduced efficacy, or an explicit recommendation to avoid the combination).\n"
    "- COMB: the drugs are clearly given or considered together in the same regimen, patient, or experiment, "
    "but the overall effect is neutral, mixed, or not clearly judged as good or bad.\n"
    "- NO_COMB: multiple drugs are mentioned but only as lists, alternatives (A or B), different arms (A vs B), "
    "background or prior regimens, or purely theoretical/unproven interactions without clear real-world "
    "coadministration and outcome.\n\n"
    "Only predict POS/NEG/COMB when co-use or interaction in this setting is explicit or very strongly implied. "
    "If you are unsure, choose NO_COMB.\n\n"
    "When you respond, you must first think step by step inside <think>...</think> in a structured way, "
    "then output the final relations as a JSON array inside <answer>...</answer>."
)


USER_PROMPT_TEMPLATE = (
    "Decide whether the following target sentence, using the context paragraph, contains any drug combination "
    "relations that should be extracted under the guideline in the system message.\n\n"
    "First, provide your reasoning inside <think>...</think> using FOUR sections:\n"
    "[1] Clinical scenario\n"
    "[2] Candidate drugs and regimen focus\n"
    "[3] Combination reasoning and clinical effect\n"
    "[4] Extraction-oriented clinical summary\n"
    "- Under each section, write only short bullet points starting with \"- \".\n"
    "- Keep the total reasoning concise, about 100–200 words.\n\n"
    "Then, immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.\n"
    "- Inside <answer>, return a valid JSON array with the form:\n"
    "<answer>\n"
    "[{{\"type\": \"POS|NEG|COMB|NO_COMB\", \"ent_set\": [\"drug1\", \"drug2\", \"...\"]}}]\n"
    "</answer>\n"
    "- Do not include any extra text or explanation outside <answer>.\n\n"
    "Target sentence: {sentence}\n"
    "Context paragraph: {paragraph}\n"
)


# Descriptive aliases for users importing this public prompt module.
ANALYST_PROMPT_TEMPLATE = RE_COT_TEMPLATE
LABEL_BLIND_ANALYST_PROMPT_TEMPLATE = RE_TEMPLATE
REVIEWER_PROMPT = evaluation_prompt

__all__ = [
    "ANALYST_PROMPT_TEMPLATE",
    "LABEL_BLIND_ANALYST_PROMPT_TEMPLATE",
    "REVIEWER_PROMPT",
    "RE_COT_TEMPLATE",
    "RE_TEMPLATE",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "evaluation_prompt",
]
