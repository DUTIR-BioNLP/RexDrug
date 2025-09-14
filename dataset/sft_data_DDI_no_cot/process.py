import json
from collections import defaultdict
input_file = ''
output_file = ''


# Function to process a JSONL dataset and save the results to a JSON file
def process_and_save_jsonl(input_file, output_file):
    # Read the input JSONL file
    with open(input_file, 'r') as infile:
        jsonl_data = [json.loads(line) for line in infile]
        for data in jsonl_data:
            data['relations'] = [rel for rel in data.get('relations', []) if rel.get('type') != 'other']
    
    processed_data = []
    CATEGORY_MAP = {
        'INT': 'INTERACTION',
        # 如有更多类别，可继续添加
    }
    for record in jsonl_data:
        context = record["context"]
        entities = [{'entity': entity['entity'], 'category': entity['category']} for entity in record["entities"]]
        grouped = defaultdict(list)
        for item in entities:
            raw_cat  = item['category']
            ent = item['entity']
            mapped_cat = CATEGORY_MAP.get(raw_cat, raw_cat) 
            if ent not in grouped[mapped_cat]:  # 去重但保留顺序
                grouped[mapped_cat].append(ent)
        entities_result = [{'type': k, 'ent': v} for k, v in grouped.items()]
        # Prepare the instruction and output
        instruction = "You are an expert in relationship extraction in the bio-medical field that provides well-reasoned and detailed responses."
        
        prompt_with_explanation = "Please identify possible drug entities from the given sentence. Then, based on the text description or relevant data, infer any existing drug-drug interaction (DDI) relationships: \n1. 'mechanism': Describes interactions between two drugs caused by pharmacokinetic (PK) mechanisms, such as interference with metabolism, absorption, or excretion. (e.g., enzyme inhibition, metabolic pathway changes). \n2. 'effect': Describes the pharmacodynamic (PD) effects or side effects resulting from the drug interaction. (e.g., increased toxicity, reduced efficacy). \n3. 'advise': Provides medical advice, warnings, or contraindications based on known or potential interactions. (e.g., avoid co-administration, monitoring required). \n4. 'int': Indicates that a drug-drug interaction exists, but does not specify the mechanism details are given. (e.g., has been established, an interaction exists).\nOutput the thinking process in <think>\n...\n</think>\n and final answer in <answer>\n...\n</answer> tags.\nOutput the relationships inside the text in JSON format.\nThe format should be as follows: <think>\n thinking process here \n</think>\n <answer>\n[{'type': 'type1', 'ent_set': ['ent1', 'ent2']},{'type': 'type2', 'ent_set': ['ent3', 'ent4']},...]\n</answer>. If no drug-drug interaction relationship is identified, output [{'type': 'NO_COMB', 'ent_set': []}]\n\n"

        output = []
        if not record['relations']:
            output.append({'type': 'NO_COMB', 'ent_set': []})
        else:
            seen = set()
            for relation in record['relations']:
                # 对实体列表排序后再转为元组，用于无序判重
                ent_key = tuple(sorted(relation['ent_sets']))
                key = (relation['type'], ent_key)
                if key not in seen:
                    seen.add(key)
                    output.append({'type': relation['type'], 'ent_set': relation['ent_sets']})

        # Converting instruction, input, output to string format
        final_output = {
            "instruction": instruction,
            "input": f"{prompt_with_explanation}\n-Sentence: {context}\n",
            "output": f"{output}",
            "the target sentence": f"{context}"
        }
        
        processed_data.append(final_output)
    
    # Save processed data to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=2)

process_and_save_jsonl(input_file, output_file)

# Output message
print(f"Processed data has been saved to {output_file}.")