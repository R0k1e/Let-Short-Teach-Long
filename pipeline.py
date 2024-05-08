from sumTree import SumTree
from Generator import Generator
import Generator_utils
from langchain_community.llms import VLLM
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
import os, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

def pipeline(generator, text, paths):
    sumTree = SumTree(text, generator)
    sumTree.info()
    Generator_utils.dumpTree(paths["treePath"], sumTree)
    node = sumTree.getRandomNode()
    questionMeta = generator.ask(node.getSummarisation())
    question =questionMeta['question']
    # answers, answerList = Generator.mr_map(generator, node.getSourceSplitText(), question)
    # answer = Generator.mr_reduce(generator, answers, question)
    answer, answerList = generator.refine(node.getSourceSplitText(), question)
    Generator_utils.dumpIntermediate(paths["mapPath"], answerList)
    Generator_utils.dump(generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)

def constructPath(model_name):
    model_name = model_name.split("/")[-1]
    path ="./outputData/"
    path = os.path.join(path, model_name)
    timeNow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    path = os.path.join(path, timeNow)
    os.makedirs(path)
    dataPath = os.path.join(path, "longContext.jsonl")
    mapPath = os.path.join(path, "map.jsonl")
    treePath = os.path.join(path, "tree.jsonl")
    paths = {"dataPath" : dataPath, "mapPath" : mapPath, "treePath" : treePath}
    Generator_utils.clear(dataPath)
    Generator_utils.clear(mapPath)
    Generator_utils.clear(treePath)
    return paths
    

if __name__ == "__main__":
    model = "/data/public/wangshuo/LongContext/model/meta-llama/Meta-Llama-3-8B-Instruct"
    # model = "gpt-3.5-turbo"
    llm = VLLM(
            model=model,
            trust_remote_code=True,  # mandatory for hf models
            tensor_parallel_size=2,
            top_k=10,
            top_p=0.95,
            temperature=0.8
    )
    # gpt = ChatOpenAI(temperature=0, model_name=model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    paths = constructPath(model)
    # vllm needs a tokenizer, gpt does not
    generator = Generator(llm, tokenizer=tokenizer)
    with open('/data/public/wangshuo/LongContext/data/LongAlignProcessed.jsonl', 'r') as file:
        for line in file:
            obj = json.loads(line)
            text = obj['text']
            pipeline(generator, text, paths)