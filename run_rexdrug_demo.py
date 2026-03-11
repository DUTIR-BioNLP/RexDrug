import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── 1. Configure paths ──────────────────────────────────────────────────────
BASE_MODEL_PATH = "DUTIR-BioNLP/RexDrug-base"       
ADAPTER_PATH    = "DUTIR-BioNLP/RexDrug-adapter"      

# ─── 2. Load model and tokenizer ─────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("Loading base model ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter ...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# ─── 3. Prepare a demo example ───────────────────────────────────────────────
# Expected answer: POS — pentostatin + cyclophosphamide + rituximab
system_msg = (
    "You are an expert in biomedical drug-drug relation extraction.\n\n"
    "Your goal is to decide whether a sentence (with its context) contains "
    "any clinically relevant drug combination relations, and to explain your reasoning.\n\n"
    "Use these relation types:\n"
    "- POS: the drugs are used together or as a regimen and the text clearly "
    "describes a beneficial effect (e.g. better response, survival, or clear synergy).\n"
    "- NEG: the drugs are used together or interact and the text clearly "
    "describes a harmful effect (e.g. more toxicity or risk, reduced efficacy, "
    "or an explicit recommendation to avoid the combination).\n"
    "- COMB: the drugs are clearly given or considered together in the same "
    "regimen, patient, or experiment, but the overall effect is neutral, mixed, "
    "or not clearly judged as good or bad.\n"
    "- NO_COMB: multiple drugs are mentioned but only as lists, alternatives "
    "(A or B), different arms (A vs B), background or prior regimens, or purely "
    "theoretical/unproven interactions without clear real-world coadministration "
    "and outcome.\n\n"
    "Only predict POS/NEG/COMB when co-use or interaction in this setting is "
    "explicit or very strongly implied. If you are unsure, choose NO_COMB.\n\n"
    "When you respond, you must first think step by step inside <think>...</think> "
    "in a structured way, then output the final relations as a JSON array inside "
    "<answer>...</answer>."
)

user_msg = (
    "Decide whether the following target sentence, using the context paragraph, "
    "contains any drug combination relations that should be extracted under the "
    "guideline in the system message.\n\n"
    "First, provide your reasoning inside <think>...</think> using FOUR sections:\n"
    "[1] Clinical scenario\n"
    "[2] Candidate drugs and regimen focus\n"
    "[3] Combination reasoning and clinical effect\n"
    "[4] Extraction-oriented clinical summary\n"
    "- Under each section, write only short bullet points starting with \"- \".\n"
    "- Keep the total reasoning concise, about 100–200 words.\n\n"
    "Then, immediately after </think>, output ONLY the final relation extraction "
    "result inside an <answer> tag.\n"
    "- Inside <answer>, return a valid JSON array with the form:\n"
    "<answer>\n"
    "[{\"type\": \"POS|NEG|COMB|NO_COMB\", \"ent_set\": [\"drug1\", \"drug2\", \"...\"]}]\n"
    "</answer>\n"
    "- Do not include any extra text or explanation outside <answer>.\n\n"
    "Target sentence: Long-term overall- and progression-free survival after "
    "pentostatin , cyclophosphamide and rituximab therapy for indolent "
    "non-Hodgkin lymphoma .\n"
    "Context paragraph: Long-term overall- and progression-free survival after "
    "pentostatin , cyclophosphamide and rituximab therapy for indolent non-Hodgkin "
    "lymphoma . In a prospective phase II trial, pentostatin combined with "
    "cyclophosphamide and rituximab (PCR) induced strong responses and was "
    "well-tolerated in previously untreated patients with advanced-stage, indolent "
    "non-Hodgkin lymphoma (iNHL). After a median patient follow-up of more than "
    "108 months, we performed an intent-to-treat analysis of our 83 participants. "
    "Progression-free survival (PFS) rates at 108 months for follicular lymphoma "
    "(FL), marginal zone lymphoma (MZL) and small lymphocytic lymphoma (SLL) were "
    "71%, 67% and 15%, respectively, and were affected by clinicopathological "
    "characteristics. Ten-year PFS rates for those with beta-2-microglobulin "
    "levels <2.2 and >=2.2 mg/l prior to treatment were 71% and 21%, respectively. "
    "Patients without bone marrow involvement had 10-year PFS rates of 72% vs. 29% "
    "for those with involvement. At time of analysis, the median overall survival "
    "(OS) had not been reached. The OS rate was 64% at 10 years and differed "
    "significantly based on histology: 94% for FL, 66% for MZL and 39% for SLL. "
    "Long-term toxicities included 18 (21.7%) patients with second malignancies "
    "and 2 (2.4%) who developed myelodysplastic syndrome after receiving additional "
    "lines of chemotherapy. Our 10-year follow-up analysis confirms that PCR is an "
    "effective, robust and tolerable treatment regimen for patients with iNHL.\n"
)

# ─── 4. Build chat messages and tokenize ──────────────────────────────────────
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user",   "content": user_msg},
]

input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# ─── 5. Run inference ────────────────────────────────────────────────────────
print("Running inference ...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

# ─── 6. Decode and display ───────────────────────────────────────────────────
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 60)
print("Model Response:")
print("=" * 60)
print(response)
print("=" * 60)