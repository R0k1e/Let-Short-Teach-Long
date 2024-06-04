import json
import random

# Path to the input JSONL file
input_file = 'data/LongAlignProcessedNumbered.jsonl'

# Path to the output file
output_file = 'data/LongAlignProcessedNumbered1k.jsonl'

# Number of lines to sample
sample_size = 1000

# Read the input JSONL file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Randomly sample lines
sampled_lines = random.sample(lines, sample_size)

# Write sampled lines to the output file
with open(output_file, 'w') as f:
    for line in sampled_lines:
        # Parse JSON from each line
        data = json.loads(line)
        
        # Write the desired data to the output file
        f.write(json.dumps(data, ensure_ascii=False) + '\n')