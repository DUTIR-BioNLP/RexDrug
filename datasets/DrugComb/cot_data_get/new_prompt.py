cot_teacher_prompt = """
You are a biomedical clinician-researcher and an expert in drug combination relation extraction.

You will be given:
- a Target Sentence,
- a Reference Context, and
- an expert-validated extraction of drug combinations and their clinical meaning (the correct RE for this case).

Your task:
Write a short, structured, clinically interpretable reasoning chain that:
(1) explains which drug entities are relevant in this case, and
(2) explains which drug combinations and clinical effects should be extracted,
in a way that is consistent with the expert-validated extraction.

Important:
- Treat the expert-validated extraction as correct constraints on the final result.
- Your reasoning should look like natural forward reasoning from the text, as if you inferred the result yourself.
- You must follow the labeling guideline below.
- Do NOT say that you were given the answer.
- Do NOT mention "gold answer", "ground truth", "labels", or "annotation" in your reasoning.

---------------- LABELING GUIDELINE ----------------
You will see relations of four types: POS, NEG, COMB, NO_COMB.

- POS:
  Label POS when the text clearly indicates that this set of drugs, used in combination
  or as a specific regimen, is associated with a beneficial clinical effect
  (e.g. improved response, survival, or synergistic activity) in the current context.

- NEG:
  Label NEG when the text clearly indicates that this drug combination or interaction
  is associated with harmful or antagonistic outcomes
  (e.g. increased toxicity or risk, reduced efficacy, or an explicit recommendation
  to avoid the combination).

- COMB:
  Label COMB when the text clearly states that these drugs are given together or considered
  as a joint regimen/interaction (e.g. part of the same chemotherapy protocol,
  coadministered, used in the same treatment course or experimental setup),
  but the overall clinical effect is neutral, mixed, or not clearly judged as good or bad.

- NO_COMB:
  Label NO_COMB when multiple drugs are mentioned but they are only listed, compared,
  or used in separate regimens/background, with no clear evidence that they act as a single
  combined therapy or interaction set in this setting.
  Typical NO_COMB patterns include: alternative treatments (A or B), different trial arms (A vs B),
  previous or other regimens not under current evaluation, and purely theoretical or unproven
  interactions without clear real-world coadministration and outcome.
----------------------------------------------------

---------------- INPUT ----------------
Target Sentence:
{TARGET_SENTENCE}

Reference Context:
{REFERENCE_CONTEXT}

Expert-validated drug combinations and clinical effects (this is the correct set you must match, in JSON-like form):
{GOLD_RE}
---------------------------------------

When you answer, follow this exact format:

1. First, output your reasoning inside <think>...</think>.

   Inside <think>, write FOUR numbered sections with bullet points ("- "):

   [1] Clinical scenario
   - 1-3 bullets.
   - Describe the main disease/condition or clinical setting.
   - State the treatment intent (e.g. induction, maintenance, prophylaxis, toxicity management).
   - Indicate whether the target sentence describes an active regimen, a comparison between regimens, a safety concern, or only background information.

   [2] Candidate drugs and regimen focus
   - 2-4 bullets.
   - Identify drugs that are potentially part of a regimen or interaction pattern in this case.
   - Emphasize drugs that appear in the expert-validated relations above, and explain why they are the core agents (e.g. explicitly named in a regimen, discussed as interacting).
   - Mention other drugs only if they play a supportive, alternative, or background role, and clarify that role.

    [3] Combination reasoning and clinical effect
    - 3-6 bullets.
    - For each relation in the expert-validated extraction, state whether the drugs are actually used or considered together in this case (or that no combination is extracted for NO_COMB).
    - Point to the key textual evidence or phrases that support your decision
        (e.g. "combination chemotherapy of A, B and C", "coadministration of A and B", "patients receiving A plus B").
    - In one short sentence per relation, summarize the clinical effect according to the guideline
        (beneficial, harmful, neutral/uncertain, or no extractable combination).
    - In NO_COMB cases, include at least one bullet that clearly explains why no drug combination
        should be extracted under the guideline (e.g. only alternatives or trial arms, theoretical or unproven interaction, separate regimens, or prior/background treatments).

   [4] Extraction-oriented clinical summary
   - 2-4 bullets.
   - Summarize, in clinician-friendly language, which drug combinations and clinical effects should be captured under the guideline.
   - Make the conclusion explicit (e.g. "no extractable combination in this sentence, so NO_COMB is appropriate" or "A and B form the key harmful interaction, labeled NEG").
   - Phrase this as a brief justification that would make sense to another clinician reading the case.

   Requirements for the content inside <think>:
   - Each section header ([1]...[4]) must be on its own line (for example: a line that only contains "[1] Clinical scenario").
   - Under each header, write only bullet points starting with "- ". Do not write free-text paragraphs.
   - Each bullet should be short (usually one sentence), clinically oriented, and specific to this case.
   - Keep the total reasoning concise: usually between 100 and 200 words, and always below 300 words.

2. Immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.
   - Inside <answer>, output a VALID JSON array, with NO extra text.
   - The JSON must follow this schema:

<answer>
[
  { "type": "POS|NEG|COMB|NO_COMB", "ent_set": ["drug1", "drug2", "..."] },
  ...
]
</answer>

Constraints:
- The JSON you output inside <answer> must be syntactically valid and directly parseable by Python json.loads.
- The content (relation types and ent_set drug names) must exactly match the expert-validated extraction given in the input (same relation types, same drugs, same spelling; you may follow the same ordering as in {GOLD_RE}).
"""
