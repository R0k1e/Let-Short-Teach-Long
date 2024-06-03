import json

def checkFailedTree(data_path):
    data_path = data_path + "/tree.jsonl"
    with open(data_path, 'r') as file:
        max_id = 0
        existing_ids = []
        with open(data_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                id = obj['id']
                existing_ids.append(id)
                max_id = max(max_id, id)
    min_id = 1
    for i in range(1, max_id + 2):
        if i not in existing_ids:
            min_id = i
            break
        
    return min_id


def checkFailedRefine(data_path, taget_num = 5):
    data_path = data_path + "/refine.jsonl"
    with open(data_path, 'r') as file:
        max_id = 0
        existing_ids = {}
        with open(data_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                id = obj['id']
                existing_ids[id] = existing_ids.get(id, 0) + 1
                max_id = max(max_id, id)
    min_id = 1
    for i in range(1, max_id + 2):
        if existing_ids.get(i, 0) < taget_num:
            min_id = i
            break

    return min_id


def checkFailedData(data_path, taget_num = 5):
    data_path = data_path + "/longContext.jsonl"
    with open(data_path, 'r') as file:
        max_id = 0
        existing_ids = {}
        with open(data_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                id = obj['id']
                existing_ids[id] = existing_ids.get(id, 0) + 1
                max_id = max(max_id, id)
    min_id = 1
    for i in range(1, max_id + 2):
        if existing_ids.get(i, 0) < taget_num:
            min_id = i
            break

    return min_id


def getData(dataPath, id):
    data = []
    with open(dataPath, 'r') as file:
        for i, line in enumerate(file):
            if i+1 == id:
                print(line)
            


if __name__ == "__main__":
    data_path = "outputData/LongAlignProcessed/2024-05-18-21-16-05"
    failed_tree = checkFailedTree(data_path)
    failed_refine = checkFailedRefine(data_path)
    failed_data = checkFailedData(data_path)
    print(failed_tree, failed_refine, failed_data)
    # path = "/data/public/wangshuo/LongContext/data/LongAlignProcessed.jsonl"
    # getData(path, failed_data)