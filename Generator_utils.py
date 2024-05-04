import json


def clear(path):
    with open(path, 'w') as file:
        file.write("")


def dump(path, text, question, answer):
    data = {
        "text": text,
        "question": question,
        "answer": answer
    }

    with open(path, 'a') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")