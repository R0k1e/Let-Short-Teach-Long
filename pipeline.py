from sumTree import SumTree
from Generator import Generator
import Generator_utils
from langchain_community.llms import VLLM, VLLMOpenAI
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
import concurrent.futures
import os, json, time, argparse
import fasttext
import tiktoken

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name or path")
parser.add_argument("--data", type=str, help="data path")
args = parser.parse_args()

lang_recognizer = fasttext.load_model('./lid.176.bin')

def identify_language(text):
        text = text.split('\n')[0]
        predictions = lang_recognizer.predict(text, k=1)  
        language_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return language_code


def pipeline(data_id, llm, tokenizer, text, paths):
    lang = identify_language(text.split('\n')[0])
    generator = Generator(llm, tokenizer, lang)
    sumTree = SumTree(text, generator, lang = lang)
    sumTree.info()
    Generator_utils.dumpTree(data_id, paths["treePath"], sumTree)
    nodes = sumTree.levelOrderTraversal()
    # node = sumTree.getRandomNode()
    for node in nodes:
        questionCategories = Generator_utils.getPrompt("questionCategory", lang = lang)
        comprehensions = Generator_utils.getPrompt("comprehension", lang = lang)
        for category in list(questionCategories.keys()):
            for comprehension in list(comprehensions.keys()):
                questionMeta={"questionCategory": category, "comprehension":comprehension}
                questionMeta = generator.ask_all_type(node.getSummarisation(),questionMeta)
                question =questionMeta['question'] 
                answer, answerList = generator.pure_refine(sumTree.getTextArray(), question)
                Generator_utils.dumpIntermediate(data_id, paths["mapPath"], answerList)
                Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)


def pipeline_randomNode(data_id, llm, tokenizer, text, path):
    lang = identify_language(text.split('\n')[0])
    generator = Generator(llm, tokenizer, lang)
    sumTree = SumTree(text, generator, lang = lang)
    sumTree.info()
    Generator_utils.dumpTree(data_id, paths["treePath"], sumTree)
    node = sumTree.getRandomNode()
    questionMeta = generator.ask(node.getSummarisation())
    question =questionMeta['question'] 
    answer, answerList = generator.pure_refine(sumTree.getTextArray(), question)
    Generator_utils.dumpIntermediate(data_id, paths["mapPath"], answerList, node)
    Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)


# no split
def pipeline_entire(data_id, llm, tokenizer, text, paths):
    lang = identify_language(text.split('\n')[0])
    generator = Generator(llm, tokenizer, lang)
    questionCategories = Generator_utils.getPrompt("questionCategory", lang = lang)
    comprehensions = Generator_utils.getPrompt("comprehension", lang = lang)
    for category in list(questionCategories.keys()):
        for comprehension in list(comprehensions.keys()):
            questionMeta={"questionCategory": category, "comprehension":comprehension}
            questionMeta = generator.ask_all_type(text,questionMeta)
            question =questionMeta['question'] 
            answer = generator.firstAnswer(text, question)
            Generator_utils.dump(data_id, generator, paths["dataPath"], text, questionMeta, answer)


def refine_hybrid(extract_generator, answer_generator, context: list[str], question, group_size = 3):
    intermediate = ""
    result = ""
    answerList = {}
    answerList["extractor"] = extract_generator.llm.model_name
    answerList["answerer"] = answer_generator.llm.model_name
    for i in range(len(context)):
        extract_result = extract_generator.extract(context[i], question)
        print(f"Chunk {i+1}/{len(context)}:\n{extract_result}")
        intermediate += extract_result
        answerList["Chunk "+str(i+1)] = context[i]
        answerList["Extracted "+str(i+1)] = extract_result
        if i % group_size == group_size - 1 or i == len(context) - 1:
            if i // group_size == 0:
                answer_generator.firstAnswer(intermediate, question)
            else:
                answer_generator.followingAnswer(intermediate, question, result)
            answerList["Intermediate input"+str(i//group_size)] = intermediate
            answerList["Intermediate result"+str(i//group_size)] = result
            print(f"Intermediate result:\n{result}")
            intermediate = ""
    return result, answerList


def constructPath(data_path):
    data_name = data_path.split("/")[-1]
    data_name = data_name.split(".")[0]
    path ="./outputData/"
    path = os.path.join(path, data_name)
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    path = os.path.join(path, timeNow)
    os.makedirs(path)
    dataPath = os.path.join(path, "longContext.jsonl")
    mapPath = os.path.join(path, "refine.jsonl")
    treePath = os.path.join(path, "tree.jsonl")
    paths = {"dataPath" : dataPath, "mapPath" : mapPath, "treePath" : treePath}
    Generator_utils.clear(dataPath)
    Generator_utils.clear(mapPath)
    Generator_utils.clear(treePath)
    return paths


def dispatcher(model, data_path):
    tokenizer = AutoTokenizer.from_pretrained(model)
    llms = []
    for i in range(2):
        llm = VLLMOpenAI(
            openai_api_key="s2l",
            openai_api_base=f"http://127.0.0.1:{3660+i}/v1",
            model_name="MiniCPM-2B-sft-bf16"
        )
        llms.append(llm)
    
    paths = constructPath(data_path)
    
    #lang
    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
        data_id = 1
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                obj = json.loads(line)
                text = obj['text']
                executor.submit(pipeline_randomNode, data_id, llms[i%2], tokenizer, text, paths)
                data_id += 1

if __name__ == "__main__":
    model = args.model
    data_path = args.data
    model = "gpt-3.5-turbo"
    llm = VLLM(
            model="../MiniCPM-2B-sft-bf16",
            trust_remote_code=True,  # mandatory for hf models
            tensor_parallel_size=1,
            top_k=10,
            top_p=0.95,
            temperature=0.8
    )
    gpt = ChatOpenAI(temperature=0, model_name=model)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
    encoding = tiktoken.encoding_for_model(model)
    paths = constructPath(data_path)
    # vllm needs a tokenizer, gpt does not
    # generator = Generator(llm, tokenizer=tokenizer)
    data_id = 1
    with open(data_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            text = obj['text']
            #lang
            pipeline_randomNode(data_id, gpt, encoding, text, paths)
            data_id += 1
    # dispatcher(model, data_path)