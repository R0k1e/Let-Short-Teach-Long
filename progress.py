import os


input_path = 'outputData/LongAlignProcessed/2024-06-06-20-00-29'
data_file = os.path.join(input_path, 'longContext.jsonl')

with open(data_file, 'r') as file:
    lines = file.readlines()
    print(len(lines))

