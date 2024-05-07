from langchain_core.documents import Document
import tiktoken
import json
from sumTree import SumTree

def clear(path):
    with open(path, 'w') as file:
        file.write("")


def dump(path, text, questionMeta, answer):
    length = tokenLength(text)+tokenLength(questionMeta["question"])+tokenLength(answer)
    data = {
        "text": text,
        "question": {
            "question": questionMeta["question"],
            "questionCategory": questionMeta["questionCategory"],
            "comprehension": questionMeta["comprehension"]
        },
        "answer": answer,
        "length": length
    }

    with open(path, 'a') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

def dumpTree(path, sumTree: SumTree):
    with open(path, 'a') as file:
        json.dump(sumTree.levelOrderForDump(), file, ensure_ascii=False)
        file.write("\n")
    

def dumpIntermediate(path, answerList):
    with open(path, 'a') as file:
        json.dump(answerList, file, ensure_ascii=False)
        file.write("\n")

def docParser(docs : list) -> list[Document]:
    documents = [Document(page_content=doc) for doc in docs]
    return documents

def tokenLength(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def getPrompt(task):
    # identifyType, metaQuestion, summarise, ask, complicate, score, answer
    with open('promptTemplate.json') as f:
        promptsLine = f.read()
        prompts = json.loads(promptsLine)
        return prompts[task]