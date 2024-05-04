from sumTree import SumTree, SumTreeNode
import json
from Generator_vllm import Generator
import Generator_utils


def test(generator, text):
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
            test(generator, text)