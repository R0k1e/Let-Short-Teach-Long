import json
import random
import os


# get total number of lines in the file and sample a subset of the data
input_path = 'outputData/LongAlignProcessed/2024-06-03-20-51-09'
data_file = os.path.join(input_path, 'longContext.jsonl')
intermediate_file = os.path.join(input_path, 'refine.jsonl')
tree_file = os.path.join(input_path, 'tree.jsonl')
output_file = 'output.json'

sample_size = 1

with open(data_file, 'r') as file:
    lines = file.readlines()
    print(len(lines))

sampled_lines = random.sample(lines, sample_size)

with open(output_file, 'w') as file:
    for line in sampled_lines:
        sampled_data = json.loads(line)
        data_id = sampled_data['id']
        with open(intermediate_file, 'r') as intermediate:
            for intermediate_line in intermediate:
                intermediate_data = json.loads(intermediate_line)
                if intermediate_data['id'] == data_id and intermediate_data['position'] == sampled_data['position']:
                    sampled_data['intermediate'] = intermediate_data
                    break
        with open(tree_file, 'r') as tree:
            for tree_line in tree:
                tree_data = json.loads(tree_line)
                if tree_data['id'] == data_id:
                    sampled_data['tree'] = tree_data
                    break
        json.dump(sampled_data, file, ensure_ascii=False)
        file.write('\n')