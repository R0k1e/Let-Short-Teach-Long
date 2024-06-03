from sumTree import SumTree
from Generatorllm import Generatorllm
import Generator_utils
from langchain_community.llms import VLLM, VLLMOpenAI
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
import concurrent.futures
import os, json, time, argparse, queue
import fasttext
import tiktoken

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name or path")
parser.add_argument("--data", type=str, help="data path")
parser.add_argument("--resume", type=bool, help="continue from last checkpoint", default=False)
parser.add_argument("--resume_path", type=str, help="resume path", default=None)
args = parser.parse_args()


def pipeline(data_id, llm, tokenizer, text, paths):
    lang = Generator_utils.identify_language(text.split('\n')[0])
    generator = Generatorllm(llm, tokenizer, lang)
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
                Generator_utils.dumpIntermediate(data_id, paths["mapPath"], answerList, node)
                Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)


def pipeline_randomNode(data_id, llm, tokenizer, text, path):
    lang = Generator_utils.identify_language(text.split('\n')[0])
    generator = Generatorllm(llm, tokenizer, lang)
    sumTree = SumTree(text, generator, chunk_size=1024, lang = lang)
    sumTree.info()
    Generator_utils.dumpTree(data_id, paths["treePath"], sumTree)
    node = sumTree.getRandomNode()
    questionMeta = generator.ask(node.getSummarisation())
    question =questionMeta['question'] 
    answer, answerList = generator.pure_refine(sumTree.getTextArray(), question)
    Generator_utils.dumpIntermediate(data_id, paths["mapPath"], answerList, node)
    Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)


def checkEqualNode(node1, node2):
    if node1.getLevel() == node2.getLevel() and node1.getIndex() == node2.getIndex():
        return True
    return False


def pipeline_Level(data_id, llms, tokenizer, text, paths):
    llm = llms.get()
    try:
        lang = Generator_utils.identify_language(text.split('\n')[0])
        generator = Generatorllm(llm, tokenizer, lang)
        sumTree = SumTree(text, generator, lang = lang, chunk_size = 2048)
        sumTree.info()
        Generator_utils.dumpTree(data_id, paths["treePath"], sumTree)
        nodes = sumTree.getNodeEachLevel()
        allNodes = sumTree.levelOrderTraversal()
        if len(allNodes) < 5: # for the line 340 which is too short with only 4 nodes. 340 causes infinite loop
            nodes = allNodes
            while len(nodes) < 5:
                node = sumTree.getRandomNode()
                nodes.append(node)
        else:
            while len(nodes) < 5:
                node = sumTree.getRandomNode()
                if not any(checkEqualNode(node, n) for n in nodes):
                    nodes.append(node)
        
        for node in nodes:
            questionMeta = generator.ask(node.getSummarisation())
            question =questionMeta['question'] 
            answer, answerList = generator.pure_refine(sumTree.getTextArray(), question)
            Generator_utils.dumpIntermediate(data_id, paths["mapPath"], question, answerList, node)
            Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)
    except Exception as e:
        raise e
    finally:
        llms.put(llm)


# no split
def pipeline_entire(data_id, llm, tokenizer, text, paths):
    lang = Generator_utils.identify_language(text.split('\n')[0])
    generator = Generatorllm(llm, tokenizer, lang)
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


def handle_exceptions(futures):
    for future in concurrent.futures.as_completed(futures.keys()):
        data_id = futures[future]
        try:
            result = future.result()
        except Exception as e:
            with open("error_log.txt", "a") as file:
                file.write(f"id:{data_id}Caught an exception: {e}\n")
            


def dispatcher(model, data_path, num_gpus = 1):
    tokenizer = AutoTokenizer.from_pretrained(model)
    llms = queue.Queue()
    for i in range(num_gpus):
        llm = VLLMOpenAI(
            openai_api_key="s2l",
            openai_api_base=f"http://127.0.0.1:{3660+i}/v1",
            model_name=model
        )
        llms.put(llm)
    
    paths = constructPath(data_path)
    
    #lang
    with concurrent.futures.ThreadPoolExecutor(max_workers = num_gpus) as executor:
        futures = {}
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                obj = json.loads(line)
                text = obj['text']
                future = executor.submit(pipeline_Level, i+1, llms, tokenizer, text, paths)
                futures[future] = i+1
                if (i + 1) % 100 == 0:
                    handle_exceptions(futures)
                    futures = {}
        handle_exceptions(futures)
        concurrent.futures.wait(futures)


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


def clearTargetFile(data_path, min_id):
    with open(data_path, 'r') as file:
        lines = file.readlines()
    with open(data_path, 'w') as file:
        for line in lines:
            obj = json.loads(line)
            if obj['id'] <= min_id:
                file.write(line)


def clearTargetFiles(data_path, min_id):
    paths = {"dataPath" : data_path + "/longContext.jsonl", "mapPath" : data_path + "/refine.jsonl", "treePath" : data_path + "/tree.jsonl"} 
    clearTargetFile(paths["dataPath"], min_id)
    clearTargetFile(paths["mapPath"], min_id)
    clearTargetFile(paths["treePath"], min_id)
    return paths
            

def resumeDispatcher(model, data_path, resumePaths, min_id, num_gpus = 1):
    tokenizer = AutoTokenizer.from_pretrained(model)
    llms = queue.Queue()
    # llms = []
    for i in range(num_gpus):
        llm = VLLMOpenAI(
            openai_api_key="s2l",
            openai_api_base=f"http://127.0.0.1:{3660+i}/v1",
            model_name=model
        )
        llms.put(llm)
    
    #lang
    with concurrent.futures.ThreadPoolExecutor(max_workers = num_gpus) as executor:
        futures = {}
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                if i+1 < min_id:
                    continue
                obj = json.loads(line)
                text = obj['text']
                future = executor.submit(pipeline_Level, i+1, llms, tokenizer, text, resumePaths)
                futures[future] = i+1
                if (i + 1) % 100 == 0 and len(futures) > 0:
                    handle_exceptions(futures)
                    futures = {}
        handle_exceptions(futures)
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    model = args.model
    data_path = args.data
    resume = args.resume
    model = "gpt-3.5-turbo"
    # llm = VLLM(
    #         model="../MiniCPM-2B-sft-bf16",
    #         trust_remote_code=True,  # mandatory for hf models
    #         tensor_parallel_size=1,
    #         top_k=10,
    #         top_p=0.95,
    #         temperature=0.8
    # )
    gpt = ChatOpenAI(temperature=0, model_name=model)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
    encoding = tiktoken.encoding_for_model(model)
    paths = constructPath(data_path)
    # vllm needs a tokenizer, gpt does not
    # generator = Generator(llm, tokenizer=tokenizer)
    # data_id = 1
    # with open(data_path, 'r') as file:
    #     for line in file:
    #         obj = json.loads(line)
    #         text = obj['text']
    #         #lang
    #         pipeline_randomNode(data_id, generator, text, paths)
    #         data_id += 1
    if resume:
        if args.resume_path is not None:
            min_id = min(checkFailedData(args.resume_path), checkFailedRefine(args.resume_path), checkFailedTree(args.resume_path))
            paths = clearTargetFiles(args.resume_path, min_id)
            resumeDispatcher(model, data_path, paths, min_id, num_gpus = 4)
            #TODO delete and resume
        else:
            raise ValueError("Please provide the path to the data file")
    else:
        dispatcher(model, data_path, num_gpus = 4)
