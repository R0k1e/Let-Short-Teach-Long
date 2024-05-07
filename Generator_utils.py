from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import tiktoken
import json

def clear(path):
    with open(path, 'w') as file:
        file.write("")


def dump(generator, path, text, questionMeta, answer):
    if isinstance(generator.llm, ChatOpenAI):
        length = len(tiktoken.tokenize(text+questionMeta["question"]+answer))
    else:
        length = len(generator.tokenizer.tokenize(text+questionMeta["question"]+answer))
    
    
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

def dumpTree(path, sumTree):
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

def getPrompt(task):
    # identifyType, metaQuestion, summarise, ask, complicate, score, answer
    with open('promptTemplate.json') as f:
        promptsLine = f.read()
        prompts = json.loads(promptsLine)
        return prompts[task]