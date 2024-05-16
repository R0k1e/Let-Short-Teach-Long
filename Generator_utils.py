from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import tiktoken
import json

def clear(path):
    with open(path, 'w') as file:
        file.write("")


def dump(data_id, generator, path, text, questionMeta, answer, node = None):

    if isinstance(generator.llm, ChatOpenAI) and "claude" not in generator.model_name:
        encoding = tiktoken.encoding_for_model(generator.model_name)
        num_tokens = len(encoding.encode(text+questionMeta["question"]+answer))
        length = num_tokens
    else:
        length = len(generator.tokenizer.tokenize(text+questionMeta["question"]+answer))
    
    
    if node is None:
        level = 0
        index = 0
    else :
        level = node.getLevel()
        index = node.getIndex()


    data = {
        "generator" : generator.model_name,
        "id": data_id,
        "position": f"Level {level} Node {index}",
        "question": {
            "question": questionMeta["question"],
            "questionCategory": questionMeta["questionCategory"],
            "comprehension": questionMeta["comprehension"]
        },
        "answer": answer,
        "length": length,
        "text": text
    }

    with open(path, 'a') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")

def dumpTree(data_id, path, sumTree):
    with open(path, 'a') as file:
        result = {"id": data_id}
        levelOrder = sumTree.levelOrderForDump()
        result.update(levelOrder)
        json.dump(result, file, ensure_ascii=False)
        file.write("\n")
    

def dumpIntermediate(data_id, path, answerList):
    with open(path, 'a') as file:
        result = {"id": data_id}
        result.update(answerList)
        json.dump(result, file, ensure_ascii=False)
        file.write("\n")

def docParser(docs : list) -> list[Document]:
    documents = [Document(page_content=doc) for doc in docs]
    return documents

def getPrompt(task, lang = 'en'):
    prompt_path = "promptTemplate.json"
    if lang == 'zh':
        prompt_path = "promptTemplate_zh.json"
    # identifyType, metaQuestion, summarise, ask, complicate, score, answer
    with open(prompt_path) as f:
        promptsLine = f.read()
        prompts = json.loads(promptsLine)
        return prompts[task]
    
def getTemplate(model):
    with open('modelTemplate.json') as f:
        templatesLine = f.read()
        templates = json.loads(templatesLine)
        return templates[model]