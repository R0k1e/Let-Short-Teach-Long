import json
import random
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--id', type=int, required=True)
args = argparser.parse_args()


# Path to the input JSONL file
input_path = 'outputData/LongAlignProcessed/2024-05-18-21-16-05'
data_file = os.path.join(input_path, 'longContext.jsonl')
intermediate_file = os.path.join(input_path, 'refine.jsonl')
tree_file = os.path.join(input_path, 'tree.jsonl')

# Path to the output file
output_file = 'output.json'

# Read the input JSONL file
with open(data_file, 'r') as file:
    lines = file.readlines()
    targetLines = []
    for line in lines:
        data = json.loads(line)
        if data['id'] == args.id:
            targetLines.append(line)
            print(data['position'])
    print(len(lines))


# Write the sampled lines to the output file
with open(output_file, 'w') as file:
    for line in targetLines:
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