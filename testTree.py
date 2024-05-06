from sumTree import SumTree, SumTreeNode
import json
from Generator_gpt import Generator
import Generator_utils


def test(generator: Generator, text):
    sumTree = SumTree(text, generator)
    print("==========Info==========")
    sumTree.info()
    print("========================")
    print("=====Traveral starts!=====")
    sumTree.levelOrderTraversal()
    print("=====Traveral finish!=====")

    # Test getNode
    print("=====Test getNode=====")
    node:SumTreeNode = sumTree.getNode(0, 0)
    if node is None:
        print("Node 0 0 is None")
    else: 
        print("Node 0 0: ")
        print(node.getSummarisation())

        # Test asking
        print("=====Test asking=====")
        question = generator.ask(node.getSummarisation())

        # Test getSourceText
        print("=====Test getSourceText=====")
        context = node.getSourceText()
        #print(context)

        # Test answering
        print("=====Test answering=====")
        answer = generator.answer(context, question)

        path = './longContext.jsonl'
        # Test dump
        root = sumTree.getRoot()
        Generator_utils.dump(path, root.getSourceText(), question, answer)
        print("=====Test mapReduce=====")
        mr_answer = generator.mapReduce(root.getSourceSplitText(), question)
        Generator_utils.dump(path, root.getSourceText(), question, mr_answer)
        print(mr_answer)
        #Generator_utils.dump(path, mr)
        print("=====Test refine=====")
        rf_answer = generator.refine(root.getSourceSplitText(), question)
        Generator_utils.dump(path, root.getSourceText(), question, rf_answer)
        print(rf_answer)
        print("=====Test mapRank=====")
        mk_answer = generator.mapRerank(root.getSourceSplitText(), question)
        Generator_utils.dump(path, root.getSourceText(), question, mk_answer)
        print(mk_answer)



if __name__ == '__main__':
    # Read the JSONL file
    text=""
    i=0
    generator = Generator()
    with open('./output_file.jsonl', 'r') as file:
        Generator_utils.clear("./longContext.jsonl")
        for line in file:
            i+=1
            # Process each line in the JSONL file
            obj = json.loads(line)
            text = obj['text']
            if i==3: test(generator, text)