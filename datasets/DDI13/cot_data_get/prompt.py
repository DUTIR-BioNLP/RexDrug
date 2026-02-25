#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDI13 Chain-of-Thought Prompts

This module contains prompt templates for:
1. CoT generation (teacher prompt) - generates reasoning chains from expert annotations
2. CoT evaluation - evaluates the quality of generated reasoning chains
"""

# =============================================================================
# COT TEACHER PROMPT
# =============================================================================
# This prompt instructs the model to generate a structured reasoning chain
# given a target sentence and expert-validated DDI relations.
# The model should produce natural forward reasoning that appears to derive
# the relations from the sentence, not reveal that the answer was given.

COT_TEACHER_PROMPT = """You are a biomedical clinician-pharmacologist and an expert in drug-drug interaction (DDI) relation extraction. You will be given:
- a Target Sentence, and
- an expert-validated extraction of interacting drug entity pairs and their DDI meaning (the correct RE for this case).

Your task:
Write a short, structured, clinically interpretable reasoning chain that:
(1) explains which drug-related entities are relevant in this sentence, and
(2) explains which entity pairs and DDI relation types should be extracted,
in a way that is consistent with the expert-validated extraction.

Important:
- Treat the expert-validated extraction as correct constraints on the final result.
- Your reasoning should look like natural forward reasoning from the sentence, as if you inferred the result yourself.
- You must follow the labeling guideline below.
- Do NOT say that you were given the answer.
- Do NOT mention "gold answer", "ground truth", "labels", or "annotation" in your reasoning.

---------------- LABELING GUIDELINE ----------------
You will see relations of five types: int, mechanism, effect, advice, NO_COMB.

- int:
  Use when the sentence states an interaction between two drug entities but does NOT clearly specify a mechanism,
  an outcome, or a concrete recommendation that fits the categories below.
  Cues: "interacts with", "interaction between", "DDI observed" without further detail.

- mechanism:
  Use when the sentence explains HOW the interaction occurs (PK/PD mechanism),
  such as enzyme/transporter inhibition or induction, altered metabolism/clearance, receptor-level antagonism, etc.
  Cues: "inhibits CYP...", "induces metabolism", "reduces clearance via...", "P-gp...".

- effect:
  Use when the sentence describes WHAT happens as a result of co-use (clinical/biological outcome),
  such as increased exposure, reduced efficacy, toxicity, QT prolongation, bleeding risk, etc., without explicit guidance.
  Cues: "increases levels/AUC of...", "decreases effect of...", "causes toxicity...".

- advice:
  Use when the sentence provides explicit clinical guidance about co-use,
  such as avoid/contraindicated, dose adjustment, monitoring, spacing administration, or caution.
  Cues: "should not be coadministered", "avoid", "contraindicated", "adjust dose", "monitor...".

- NO_COMB:
  Use when multiple drug entities are mentioned but no interaction relation is asserted between a specific pair in this sentence.
  Typical patterns: lists, alternatives (A or B), comparisons (A vs B), separate regimens without interaction claim.
----------------------------------------------------

---------------- INPUT ----------------
Target Sentence: {TARGET_SENTENCE}
Expert-validated interacting entity pairs and DDI relation types (this is the correct set you must match, in JSON-like form): {GOLD_RE}
---------------------------------------

When you answer, follow this exact format:

1. First, output your reasoning inside <think>...</think>.
   Inside <think>, write FOUR numbered sections with bullet points ("- "):

   [1] Clinical / pharmacological scenario
   - 1-3 bullets.
   - Describe the setting implied by the sentence (e.g., coadministration warning, PK interaction statement, trial observation).
   - State whether the sentence asserts an interaction, explains a mechanism, reports an effect, gives advice, or is only background.

   [2] Candidate entities and pair focus
   - 2-4 bullets.
   - Identify drug-related entities (drug/brand/group/drug_n) that are candidates for a DDI relation in this sentence.
   - Emphasize entities that appear in the expert-validated relations and explain why they form the focal interacting pair(s).
   - Mention other entities only if they are clearly non-focal (e.g., comparators/background) and clarify that role.

   [3] Interaction reasoning and relation labeling
   - 3-6 bullets.
   - For each relation in the expert-validated extraction:
     - Confirm the sentence supports a specific entity pair being linked (or supports NO_COMB when no interaction is stated).
     - Point to the key phrase(s) that justify the chosen relation type (int vs mechanism vs effect vs advice).
     - Summarize in one short sentence per pair what is being asserted (interaction existence, mechanism, effect, or guidance).
   - In NO_COMB cases, include at least one bullet explicitly explaining why no DDI relation is extractable
     (e.g., only list/alternatives/comparison/separate regimens; no interaction claim for that pair).

   [4] Extraction-oriented clinical summary
   - 2-4 bullets.
   - State exactly which entity pairs should be extracted and their relation types under the guideline.
   - Make the conclusion explicit (e.g., "A-B is advice: avoid coadministration" or "A-B is mechanism: CYP inhibition").
   - Keep the summary aligned with what will be returned in the final JSON.

Requirements for the content inside <think>:
- Each section header ([1]...[4]) must be on its own line (a line that only contains that header).
- Under each header, write only bullet points starting with "- ". Do not write free-text paragraphs.
- Each bullet should be short, clinically/pharmacologically oriented, and specific to this sentence.
- Keep the total reasoning concise: usually between 100 and 200 words, and always below 300 words.

2. Immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.
   - Inside <answer>, output a VALID JSON array, with NO extra text.
   - The JSON must follow this schema (keep the same outer format as the original instruction):

<answer>
[{{"type": "int|mechanism|effect|advice|NO_COMB", "ent_set": ["ent1", "ent2"] }},...]
</answer>

Constraints:
- The JSON you output inside <answer> must be syntactically valid and directly parseable by Python json.loads.
- The content (relation types and ent_set entity strings) must exactly match the expert-validated extraction given in the input
  (same relation types, same entity strings, same spelling/casing; ordering may follow {GOLD_RE}).
- Do not normalize entity names; use the exact entity strings as provided/validated.
"""


# =============================================================================
# EVALUATION PROMPT
# =============================================================================
# This prompt instructs an evaluator model to score the quality of a generated
# reasoning chain across six dimensions: format, medical accuracy, semantic
# accuracy, RE consistency, naturalness, and usefulness.

EVALUATION_PROMPT = """You are a biomedical clinician-pharmacologist and an impartial reviewer.

Your task is to evaluate the quality of a reasoning chain and its relation extraction output
for a sentence-level drug-drug interaction (DDI) extraction case (DDI13-style).

You will be given:
- Target Sentence
- Expert-validated DDI relations (GOLD_RE)
- Model Reasoning (inside <think>...</think>)
- Model Answer (inside <answer>...</answer>, as JSON)

Please evaluate the reasoning and answer along SIX criteria.
Each criterion is scored from 0 to 5 (0 = hard violation, 1 = very poor, 5 = excellent).

1. score_format (0-5)
- Check if the output follows the required structure:
  - Reasoning is enclosed in <think>...</think>.
  - There are four sections labeled [1], [2], [3], [4], each header on its own line.
  - Under each section, the content is mostly written as bullet points starting with "- ".
  - The answer is enclosed in <answer>...</answer>.
  - The content inside <answer> is a valid JSON array that can be parsed by json.loads.
  - Each item should follow the schema: {{"type": "...", "ent_set": ["ent1","ent2"]}}.
- If the JSON is invalid or the overall structure is severely broken, assign 0.

2. score_medical (0-5)
- Is the reasoning pharmacologically/clinically plausible and consistent with biomedical knowledge?
- Are there serious errors about interaction mechanisms (PK/PD), interaction outcomes, or safe-use advice?
- If there are major biomedical errors that would mislead DDI interpretation, assign 0 or 1.

3. score_semantic (0-5)
- Does the reasoning accurately reflect the meaning of the Target Sentence (and only what is supported by it)?
- Avoid hallucinating details such as study design, patient population, doses, or outcomes that are not stated.
- If it contradicts the sentence or invents key facts to justify relations, assign a low score.

4. score_re_consistency (0-5)
- Compare the Model Answer JSON with GOLD_RE:
  - If they do not match (wrong relation types, missing/extra pairs, wrong entity strings), assign 0.
  - Entity strings in ent_set must match exactly (spelling/casing) and represent a binary pair.
- Check whether the reasoning inside <think> supports these exact relations:
  - The described entity pairs and the asserted interaction meaning should align with GOLD_RE.
  - The reasoning should not primarily argue for a different relation type than GOLD_RE.
- For the relation type in GOLD_RE:
  - mechanism: reasoning should explain a mechanism cue (e.g., enzyme/transporter inhibition/induction, altered metabolism/clearance, PD antagonism).
  - effect: reasoning should describe an outcome cue (e.g., increased exposure, decreased efficacy, toxicity/risk) without explicit clinical instruction.
  - advice: reasoning should describe explicit guidance (avoid/contraindicated/adjust dose/monitor/spacing).
  - int: reasoning should justify that an interaction is stated but not further specified into mechanism/effect/advice.
  - NO_COMB: reasoning should explicitly justify why no interaction relation is extractable for the mentioned entities
    (e.g., list/alternatives/comparison/separate regimens; no interaction claim).
- Higher scores mean both the JSON and the reasoning are well-aligned with GOLD_RE.

5. score_naturalness (0-5)
- Does the reasoning read like natural forward reasoning, as if the model inferred the relations from the sentence?
- If the reasoning explicitly mentions or clearly implies that it was given the answer,
  or uses terms like "gold answer", "ground truth", "labels", "annotation", assign 0.
- Penalize meta-comments about the dataset, evaluation process, or correctness that break naturalness.

6. score_usefulness (0-5)
- For an information extraction and clinical/pharmacology audience:
  - Does the reasoning point to concrete textual cues (trigger phrases) that justify the extracted relation type?
  - Does it clearly differentiate among int vs mechanism vs effect vs advice (or justify NO_COMB)?
  - Does it keep entity focus clear (drug/brand/group/drug_n mentions) without unnecessary speculation?
- Higher scores indicate that the reasoning would help:
  - (a) a model learn which textual patterns map to the DDI13 relation types, and
  - (b) a clinician understand why the specific entity pair is extracted with that DDI meaning.

Length consideration:
- The reasoning inside <think> should ideally be concise (roughly 80-220 words).
- If it is extremely long or extremely short and uninformative, you may reduce score_format or score_usefulness.

Output:
Return your evaluation as a single JSON object with the following fields:

{{
  "score_format": int,
  "score_medical": int,
  "score_semantic": int,
  "score_re_consistency": int,
  "score_naturalness": int,
  "score_usefulness": int,
  "comment": "A short summary comment (1-3 sentences)."
}}
"""
