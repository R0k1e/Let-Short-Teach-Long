import random

with open("outputData/LongAlignProcessedNumbered1k/2024-07-01-15-26-31/LongQA.jsonl", "r") as f:
    lines = f.readlines()
    line = random.sample(lines, 1)
    with open("output_qa.json", "w") as wf:
        wf.writelines(line)