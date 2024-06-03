from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize
import tiktoken
import json
import fasttext
import re
import threading


lang_recognizer = fasttext.load_model('./lid.176.bin')
file_lock = threading.Lock()

def clear(path):
    with open(path, 'w') as file:
        file.write("")


def dump(data_id, generator, path, text, questionMeta, answer, node = None):

    
    length = len(generator.tokenizer.encode(text+questionMeta["question"]+answer))
    
    
    if node is None:
        level = 0
        index = 0
    else :
        level = node.getLevel()
        index = node.getIndex()


    data = {
        "id": data_id,
        "generator" : generator.model_name,
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
        with file_lock:
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")

def dumpTree(data_id, path, sumTree):
    with open(path, 'a') as file:
        result = {"id": data_id}
        levelOrder = sumTree.levelOrderForDump()
        result.update(levelOrder)
        with file_lock:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")
    

def dumpIntermediate(data_id, path, question, answerList, node):
    with open(path, 'a') as file:
        result = {"id": data_id}
        result["question"] = question
        result["position"] = f"Level {node.getLevel()} Node {node.getIndex()}"
        result.update(answerList)
        with file_lock:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")

# def docParser(docs : list) -> list[Document]:
#     documents = [Document(page_content=doc) for doc in docs]
#     return documents

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
    

def identify_language(text):
        text = text.split('\n')[0]
        predictions = lang_recognizer.predict(text, k=1)  
        language_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return language_code


def sent_tokenize_zh(text):
    resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    s = text
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def split_text_on_tokens(text: str, tokenizer, tokens_per_chunk, chunk_overlap = 0) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokens_per_chunk - chunk_overlap
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


def splitTextOnSentences(text: str, tokenizer, lang = 'en', max_chunk_length = 2048):
    if lang == 'zh':
        sents = sent_tokenize_zh(text)
    else:
        sents = sent_tokenize(text)
    processed_sents = []
    for sent in sents:
        while len(tokenizer.encode(sent)) > max_chunk_length:
            texts = split_text_on_tokens(sent, tokenizer, max_chunk_length)
            processed_sents.append(texts[0])
            sent = ''.join(texts[1:])
        if sent:  # This ensures we add the last chunk if it's not empty
            processed_sents.append(sent)
    # Chunk sentences
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sent in processed_sents:
        if len(tokenizer.encode(current_chunk+sent)) >= max_chunk_length:
            chunks.append(current_chunk)
            current_chunk = ""
            current_length = 0
        current_chunk += sent + " "
        current_length += len(tokenizer.encode(sent))
    
    if current_chunk != "":
        chunks.append(current_chunk)
    
    return chunks
