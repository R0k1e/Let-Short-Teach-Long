import json

def get_target_line(jsonl_file, target_line_number):
    with open(jsonl_file, 'r') as file:
        lines = file.readlines()
        target_line = json.loads(lines[target_line_number])
    return target_line

def write_to_json(target_line, json_file):
    with open(json_file, 'w') as file:
        json.dump(target_line, file)

# Example usage
jsonl_file = 'outputData/LongAlignProcessed/2024-05-15-00-16-07/longContext.jsonl'
json_file = 'targetLine.json'
target_line_number = 415

target_line = get_target_line(jsonl_file, target_line_number)
write_to_json(target_line, json_file)