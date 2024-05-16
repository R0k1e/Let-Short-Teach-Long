import json

LongText = []
def write_jsonl_file(file_path, messages):
    with open(file_path, 'w') as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + '\n')


def read_jsonl_file(file_path):
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            index = data["messages"][0]["content"].rfind("\n\n")
            question = data["messages"][0]["content"][index:].strip()
            text= data["messages"][0]["content"][:index]
            LongText.append({"text":text,"question":question})

# Replace 'file_path' with the actual path to your JSONL file
file_path = '../data/LongAlign-10k/long.jsonl'
read_jsonl_file(file_path)
# Replace 'output_file_path' with the actual path to your output JSONL file
output_file_path = './output_file2.jsonl'
write_jsonl_file(output_file_path, LongText)