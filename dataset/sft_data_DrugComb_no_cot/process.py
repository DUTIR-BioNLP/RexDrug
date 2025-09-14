import json

input_file = 'final_test_set.jsonl'
output_file = 'test.json'


# Function to process a JSONL dataset and save the results to a JSON file
def process_and_save_jsonl(input_file, output_file):
    # Read the input JSONL file
    with open(input_file, 'r') as infile:
        jsonl_data = [json.loads(line) for line in infile]
    
    processed_data = []
    for record in jsonl_data:
        sentence = record["sentence"]
        paragraph = record["paragraph"]
        # You can expand this list with more relation types if needed
        entities = [entity['text'] for entity in record["spans"]]
        entities = list(set(entities))
        spans = {span["span_id"]: span["text"] for span in record["spans"]}
        # Prepare the instruction and output
        instruction = "You are an expert in drug combination relationship extraction in the medical field. Please identify possible drug combinations from the given sentence and determine their combined usage effects. You can refer to the context in the paragraph to assist with reasoning. You need to follow these steps:\n 1. Drug Entity Recognition (NER): Extract all drug entities from the 'Target Sentence' and list them.\n 2. Drug Combination Identification: Identify the drug combinations in the 'Target Sentence' using the extracted drug entities.\n 3. Relationship Judgment: If a drug combination is identified, infer the usage effect based on drug efficacy descriptions, interactions, or related numerical values: \n'POS': The drug combination has a positive effect. \n'NEG': The drug combination has a negative effect. \n'COMB': The drug combination has no significant positive or negative effect. \n 4. Output Format: Return the output in the following format:  `@ner# ['drug1', 'drug2', ...] #ner@  [{'type': 'relation_type', 'ent_set': ['drug1', 'drug2', ...]}]`.  First, list the identified drug entities within the `@ner# ... #ner@` tags. Then, provide the relationship extraction result as a list of dictionaries. If no drug combination is identified, output `@ner# ['drug1', 'drug2', ...] #ner@` followed by `['NO_COMB']`."
        
        rel_output = []
        if record['rels'] == []:
            rel_output.append("NO_COMB")
        else:
            
            for relation in record['rels']:
                ent_sets = []
                for span_id in relation['spans']:
                    ent_sets.append(spans[span_id])
                rel_output.append({'type': relation['class'], 'ent_set': ent_sets})

        output = f'@ner# {entities} #ner@ {rel_output}'
        # Converting instruction, input, output to string format
        final_output = {
            "instruction": instruction,
            "input": f"-Target Sentence: {sentence}\n-Paragraph:{paragraph}",
            "output": f"{output}"
        }
        
        processed_data.append(final_output)
    
    # Save processed data to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=2)

process_and_save_jsonl(input_file, output_file)

# Output message
print(f"Processed data has been saved to {output_file}.")