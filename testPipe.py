from sumTree import *
from Generator_vllm import Generator
import Generator_utils
import json
from concurrent.futures import ThreadPoolExecutor

def test(generator, text):

    sumTree = SumTree(text, generator)
    sumTree.info()
    root = sumTree.getRoot()
    questionMeta = Generator.ask(generator, root.getSummarisation())
    question =questionMeta['question']
    answer, answerList = Generator.mapReduceNew(generator, root.getSourceSplitText(), question)
    Generator_utils.dumpTree("./treeLog2.jsonl", sumTree)
    Generator_utils.dumpIntermediate("./mapLog2.jsonl", answerList)
    Generator_utils.dump("./longContext2.jsonl", root.getSourceText(), questionMeta, answer)
    node = sumTree.getNode(2, 0)
    questionMeta = Generator.ask(generator, node.getSummarisation())
    question =questionMeta['question']
    answer, answerList = Generator.mapReduceNew(generator, node.getSourceSplitText(), question)
    #Generator_utils.dumpTree("./treeLog2.jsonl", sumTree)
    Generator_utils.dumpIntermediate("./mapLog2.jsonl", answerList)
    Generator_utils.dump("./longContext2.jsonl", node.getSourceText(), questionMeta, answer)
    

if __name__ == "__main__":
    text=""
    i=0
    generator = Generator()
    #executor = ThreadPoolExecutor(max_workers=2)
    with open('./output_file2.jsonl', 'r') as file:
        #Generator_utils.clear("./longContext.jsonl")
        for line in file:
            i+=1
            # Process each line in the JSONL file
            obj = json.loads(line)
            text = obj['text']
            test(generator, text)