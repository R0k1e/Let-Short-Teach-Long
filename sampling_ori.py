with open("/data/public/wangshuo/LongContext/data/sft/LongAlign-10k/long.jsonl", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i+1 == 6803:
            with open("sampled.jsonl", "a") as af:
                af.write(line)
