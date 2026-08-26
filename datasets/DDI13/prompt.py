"""Prompts used for DDI13 reasoning distillation and model training.

The dual-agent pipeline uses ``RE_COT_TEMPLATE`` for the medical reasoning
analyst and ``evaluation_prompt`` for the expert reviewer. The accepted traces
are converted to SFT/GRPO chat records with ``SYSTEM_PROMPT`` and
``USER_PROMPT_TEMPLATE``.
"""

VALID_RELATION_TYPES = ("int", "mechanism", "effect", "advise", "NO_COMB")
SECTION_TITLES = (
    "[1] Sentence-level pharmacological scenario",
    "[2] Candidate entities and pair focus",
    "[3] DDI evidence and relation classification",
    "[4] Extraction-oriented clinical summary",
)
MAX_REASONING_WORDS = 350


evaluation_prompt = """
You are an expert in biomedical science and clinical pharmacology, acting as a rigorous and
impartial reviewer.

Evaluate the reasoning and relation-extraction answer for one sentence-level DDI13 case. You receive a
Target Sentence, the expert-validated name-level relations (GOLD_RE), reasoning inside
<think>...</think>, and a JSON answer inside <answer>...</answer>.

Task rules:
- Compare entity strings exactly; only pair order and relation-array order may change.
- Different types for the same name pair remain separate relations.
- A pair whose two names normalize to the same value, or a reconstructed discontinuous name, is
  valid only when present in GOLD_RE. The supplied strings must still be preserved exactly.
- Valid outputs are int, mechanism, effect, advise, and NO_COMB.

Score each criterion from 0 to 5:
5: fully correct, complete, concise, and text-grounded.
4: reliable with only minor issues that do not affect the evidence, pair, type, or conclusion.
3: usable but requiring substantive revision.
2: important problems affect the main analysis.
1: severe errors make most of the reasoning unreliable.
0: unusable or a hard constraint is violated.

1. score_format
- Output must contain exactly one <think> block followed by exactly one <answer> block, with no
  outside text.
- <think> must contain these exact headings in order, each on its own line:
  [1] Sentence-level pharmacological scenario
  [2] Candidate entities and pair focus
  [3] DDI evidence and relation classification
  [4] Extraction-oriented clinical summary
- Every other non-empty line in <think> must begin with "- ".
- <answer> must be a standard JSON array parseable by json.loads.
- A DDI item must be {"type":"...","ent_set":["entity1","entity2"]} with two non-empty strings.
- NO_COMB must be exactly [{"type":"NO_COMB","ent_set":[]}].
- Invalid labels, duplicate normalized relation keys, or mixing NO_COMB with DDI relations make
  score_format 0.

2. score_extraction
- Only 0 or 5 is allowed.
- Give 5 only when the answer exactly matches GOLD_RE after ignoring pair order and relation-array
  order. Do not change spelling, capitalization, singular/plural form, or any supplied surface form.
- Missing, added, duplicated, relabeled, or merged relations make this score 0.

3. score_semantic
- Interpret the Target Sentence independently before judging the reasoning. No reference context is
  available, so every claim must have direct support in that sentence.
- Check whether the reasoning identifies the local wording that connects each specific entity pair.
  Several drug names in one sentence do not imply an interaction between every possible pair.
- int: the sentence explicitly states a drug-drug interaction but gives no information specific
  enough for mechanism, effect, or advise. Words such as "interaction" or "interacts" may signal
  this relation, but int is not a fallback label for uncertainty.
- mechanism: the sentence explains how an interaction occurs or describes a pharmacokinetic alteration,
  including absorption, distribution, metabolism, clearance, enzyme or transporter activity, drug
  concentration, exposure, AUC, peak concentration, or half-life.
- effect: the sentence describes a clinical, toxic, pharmacodynamic, biological, behavioral,
  therapeutic, or experimental consequence, such as bleeding, hypotension, enhanced sedation or
  toxicity, reduced efficacy, or enhanced activity, without an explicit management instruction.
- A change in anesthetic requirement, efficacy, toxicity, physiological response, cellular signaling,
  phosphorylation, or another observed outcome is not a pharmacokinetic alteration merely because it
  is dose-dependent or time-dependent. Calling it pharmacokinetic without an explicit process or
  parameter makes score_semantic and score_medical at most 2. For this violation, 3 or higher is not
  allowed even when the answer matches GOLD_RE.
- Preserve the evidence setting. An in vitro, cultured-cell, ex vivo, or animal result must not be
  described as clinically relevant, a patient benefit or risk, or a clinical management implication
  unless the sentence explicitly makes that clinical connection. Such translational overstatement
  makes score_semantic and score_medical at most 2.
- advise: the sentence gives an explicit management instruction, such as avoiding coadministration,
  stating that concomitant use is contraindicated, exercising caution, monitoring, adjusting a
  dose, or separating administration times. A risk statement without an instruction is usually not
  advise.
- NO_COMB: no extractable binary DDI is asserted. This includes a sentence with no drug or only one
  drug, as well as multiple drugs that are merely listed, compared, presented as alternatives,
  discussed separately, or mentioned together without an interaction claim.
- Penalize invented mechanisms, directions of effect, patient details, doses, or outcomes. If the
  reasoning contradicts the sentence or invents evidence, this score must not exceed 2.
- If the reasoning uses a GOLD_RE type that the Target Sentence cannot justify without adding an
  unstated mechanism, outcome, instruction, or clinical interpretation, penalize the unsupported
  reasoning and identify the evidence gap in comment. Do not propose or output a replacement label.

4. score_medical
- Check the pharmacokinetic and pharmacodynamic interpretation, direction of change, affected drug,
  evidence setting, risk or monitoring target, and dose-adjustment target.
- Do not demand external facts. A serious, misleading medical error scores 0 or 1.

5. score_naturalness
- Reasoning must read as forward analysis from the sentence and must not reveal prior knowledge of
  GOLD_RE or an answer to match.
- Penalize conclusion-first rationalization, mechanical repetition, and padding. Clear disclosure of
  the private answer anchor scores 0.

6. score_reasoning
- Identify relevant entities, local triggers, extracted pairs, and why each trigger maps to its type.
- Cover every final relation without inventing unsupported pair combinations.
- For repeated names or different types on one name pair, distinguish the supporting local clauses.
- For NO_COMB, state the concrete reason no binary DDI is extractable.
- <think> should usually contain 100-220 English words and must not exceed 350 words. More than 350
  words makes score_reasoning 0.

Return only one standard JSON object:
{
  "score_format": 0,
  "score_extraction": 0,
  "score_semantic": 0,
  "score_medical": 0,
  "score_naturalness": 0,
  "score_reasoning": 0,
  "comment": "A concise overall review in 1-3 sentences."
}

Determine the sentence meaning independently before using GOLD_RE to check completeness. Never give
a high score solely because the answer matches GOLD_RE.
"""


RE_COT_TEMPLATE = """You are a biomedical researcher specializing in clinical pharmacology and
sentence-level DDI13 relation extraction.

You receive one Target Sentence and its expert-validated name-level relation set (GOLD_RE). Write a
short, structured reasoning chain that identifies relevant drug names, locates the textual evidence
for each binary pair, maps that evidence to a DDI type, and returns the exact relation set.

Requirements:
- Write the reasoning and answer in English. Preserve GOLD_RE labels and entity strings exactly.
- Use GOLD_RE only as a private answer anchor for pairs, types, and completeness; it is not evidence.
- Ground every claim in the Target Sentence. Do not invent mechanisms, outcomes, instructions,
  doses, populations, or study details.
- Keep pharmacokinetic processes distinct from pharmacodynamic or experimental outcomes. Dose or
  time dependence alone does not establish a pharmacokinetic mechanism.
- Preserve the stated evidence level. Do not turn an in vitro, cultured-cell, ex vivo, or animal
  observation into a claim of clinical relevance, patient benefit or harm, or management guidance.
- Write natural forward reasoning and never mention GOLD_RE, annotations, a supplied label, or an
  answer to match.

Name-level policy:
- A relation key consists of the relation type and an unordered pair of normalized entity names.
- Output each key once even when the source contains multiple mentions of the same names.
- Keep different relation types for one name pair when GOLD_RE contains them.
- When GOLD_RE contains a pair whose two names normalize to the same value, preserve both supplied
  strings rather than collapsing the pair.
- Preserve reconstructed discontinuous names exactly, even when they are not contiguous substrings.
- Co-occurring drug names do not imply all pairwise DDIs; use only explicitly connected pairs.

Labels:
- int: the sentence explicitly states an interaction but gives no specific mechanism, consequence,
  or management instruction. An explicit interaction statement is required; int is not a fallback
  for uncertainty.
- mechanism: the sentence explains how the interaction occurs or how one drug changes another
  drug's pharmacokinetics. Typical evidence includes altered absorption, distribution, metabolism,
  or clearance; enzyme or transporter inhibition/induction; and changes in concentration, exposure,
  AUC, peak concentration, or half-life. Do not call an observed pharmacodynamic or experimental
  outcome pharmacokinetic unless the sentence explicitly states a pharmacokinetic process or
  parameter.
- effect: the sentence describes a clinical, toxic, pharmacodynamic, biological, behavioral,
  therapeutic, or experimental consequence, such as bleeding, hypotension, enhanced sedation or
  toxicity, reduced efficacy, inhibited activity, or enhanced activity. A consequence without an
  explicit management instruction should generally be classified as effect. Dose-dependent or
  time-dependent changes in anesthetic requirement, efficacy, toxicity, cellular signaling, or
  phosphorylation remain observed effects unless the sentence provides a more specific mechanism.
- advise: the sentence gives an explicit clinical management instruction, such as avoiding
  coadministration, stating that concomitant use is contraindicated, exercising caution, monitoring,
  adjusting a dose, or separating administration times. A risk statement without an instruction is
  usually not advise.
- NO_COMB: the sentence asserts no extractable binary DDI. This includes no drug or only one drug,
  as well as multiple drugs that are merely listed, compared, presented as alternatives, discussed
  in separate monotherapy statements, mentioned as background, or simply mentioned together.

Target Sentence:
{TARGET_SENTENCE}

GOLD_RE (valid JSON; exact deduplicated set to return):
{GOLD_RE}

Output exactly one <think>...</think> followed by one <answer>...</answer>.

Inside <think>, use these exact headings in order, each on its own line. Every other non-empty line
must begin with "- ". Keep the reasoning concise, usually 100-220 words and never over 350 words.

[1] Sentence-level pharmacological scenario
- Write 1-2 bullets.
- Briefly describe the pharmacological, clinical treatment, or experimental setting.
- Preserve the evidence level: identify in vitro, cellular, ex vivo, animal, and clinical settings
  accurately, without translating an experimental finding into a clinical claim.
- State whether the sentence primarily gives background, an interaction statement, a
  pharmacokinetic mechanism or change, a clinical or experimental consequence, or an explicit
  management instruction, and whether an extractable DDI exists.

[2] Candidate entities and pair focus
- Usually write 2-4 bullets; use more when the sentence contains many entities.
- Identify the drug names directly relevant to the DDI decision, preserving their supplied English
  strings.
- State which names are connected in the same local statement by coadministration, an interaction,
  a pharmacokinetic change, a clinical or experimental consequence, or a management instruction.
- Mention other drug names only when needed to explain that they are listed, compared, alternatives,
  or background rather than part of an extracted relation.
- When one name occurs repeatedly, use the relevant positions to analyze evidence but report the
  deduplicated name-level result.

[3] DDI evidence and relation classification
- Usually write 3-6 bullets; use more for multiple pairs or relation types.
- For each candidate pair, identify the exact trigger or key phrase showing an interaction,
  coadministration, pharmacokinetic change, clinical/experimental consequence, or management
  instruction.
- Distinguish an observed pharmacodynamic or experimental outcome from a pharmacokinetic mechanism.
  Dose or time dependence alone is not pharmacokinetic evidence; name the explicit process or
  parameter before using that description.
- Keep experimental conclusions within the stated model. Do not call an in vitro, cellular, ex vivo,
  or animal finding clinically relevant unless the sentence itself provides that clinical link.
- Explain the classification of every final relation:
  - int: identify the explicit interaction statement and the absence of a more specific mechanism,
    consequence, or management instruction;
  - mechanism: identify the interaction process or pharmacokinetic alteration and the affected drug;
  - effect: identify the clinical, toxic, pharmacodynamic, or experimental consequence;
  - advise: identify the explicit recommendation to avoid coadministration, monitor, adjust a dose,
    exercise caution, or separate administration times;
  - NO_COMB: explain why the names are only listed, compared, alternatives, described separately, or
    co-occurring without an extractable DDI.
- If one name pair has different types, explain the local evidence for each type separately. Parallel
  pairs may share one concise explanation only when it clearly names every covered pair.
- DDI13 extracts binary relations only. When three or more drugs appear, identify the supported
  binary pairs rather than combining them into one higher-order entity set.

[4] Extraction-oriented clinical summary
- Write 2-4 bullets; group relations concisely when the sentence contains many pairs.
- Summarize every extracted binary name pair and relation type in biomedically precise language that
  is understandable to clinicians. Clinician-readable wording does not make experimental evidence
  clinically validated.
- State an explicit conclusion, for example that a warfarin-aspirin bleeding risk should be
  classified as effect, or that a comparison between drug names expresses no extractable DDI and
  therefore warrants NO_COMB.
- Entity strings, relation types, and relation count must exactly match <answer>, and each name-level
  relation key must be summarized once.

Inside <answer>, return only a standard JSON array:
<answer>
[{{"type":"int|mechanism|effect|advise|NO_COMB","ent_set":["entity1","entity2"]}},...]
</answer>

Answer constraints:
- Use double-quoted JSON directly parseable by json.loads.
- Each DDI item has exactly two non-empty entity strings matching GOLD_RE exactly.
- Output each normalized relation key once; retain different types for the same pair.
- With no DDI, output exactly [{{"type":"NO_COMB","ent_set":[]}}].
- Never mix NO_COMB with a DDI relation.
"""


# Prompts used in the final SFT/GRPO chat records.
SYSTEM_PROMPT = """You are an expert in biomedical drug-drug interaction relation extraction for DDI13.

Your goal is to decide whether a target sentence contains any binary drug-drug interactions, identify the drug pairs, classify each relation, and explain your reasoning.

Use these relation types:
- int: the sentence explicitly states an interaction but gives no specific mechanism, consequence, or management instruction.
- mechanism: the sentence describes how the interaction occurs or how one drug changes another drug's pharmacokinetics, such as its metabolism, clearance, concentration, or exposure.
- effect: the sentence describes a clinical, toxic, pharmacodynamic, biological, therapeutic, or experimental consequence of the interaction.
- advise: the sentence gives an explicit management instruction, such as avoiding coadministration, using caution, monitoring, adjusting a dose, or separating administration times.
- NO_COMB: no binary DDI is expressed; drugs may only be listed, compared, presented as alternatives, described separately, or mentioned together.

Only extract relations supported by the target sentence. Drug co-occurrence alone does not imply a DDI. Output binary pairs only, retain different supported relation types for the same pair, and reconstruct complete discontinuous drug names when needed. If no explicit interaction evidence is present, choose NO_COMB.

When you respond, first reason inside <think>...</think> using the required four sections, then output the final relations as a JSON array inside <answer>...</answer>."""


USER_PROMPT_TEMPLATE = """Decide whether the following target sentence contains any DDI13 drug-drug interaction relations under the guideline in the system message.

First, provide your reasoning inside <think>...</think> using FOUR sections:
[1] Sentence-level pharmacological scenario
[2] Candidate entities and pair focus
[3] DDI evidence and relation classification
[4] Extraction-oriented clinical summary
- Under each section, write only short bullet points starting with "- ".
- Base the reasoning only on the target sentence and explain the evidence for each final relation.
- Keep the total reasoning concise, about 100-220 English words and never over 350 words.

Then, immediately after </think>, output ONLY the final relation extraction result inside an <answer> tag.
- Inside <answer>, return a valid JSON array with the form:
<answer>
[{{"type":"int|mechanism|effect|advise|NO_COMB","ent_set":["drug1","drug2"]}},...]
</answer>
- Each DDI relation must contain exactly two drug names; output each pair and relation type only once.
- If there is no DDI, output exactly [{{"type":"NO_COMB","ent_set":[]}}] and do not mix NO_COMB with other relations.
- Do not include any extra text outside <think> and <answer>.

Target sentence: {sentence}
"""


# Descriptive aliases for users importing this public prompt module.
ANALYST_PROMPT_TEMPLATE = RE_COT_TEMPLATE
REVIEWER_PROMPT = evaluation_prompt

__all__ = [
    "ANALYST_PROMPT_TEMPLATE",
    "MAX_REASONING_WORDS",
    "REVIEWER_PROMPT",
    "RE_COT_TEMPLATE",
    "SECTION_TITLES",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "VALID_RELATION_TYPES",
    "evaluation_prompt",
]
