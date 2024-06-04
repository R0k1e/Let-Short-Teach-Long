import json

def add_line_numbers(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in, start=1):
            data = {}
            data['origin_id'] = i
            data.update(json.loads(line))
            f_out.write(json.dumps(data, ensure_ascii= False) + '\n')

# Usage example
input_file = 'data/LongAlignProcessed.jsonl'
output_file = 'data/LongAlignProcessedNumbered.jsonl'
add_line_numbers(input_file, output_file)