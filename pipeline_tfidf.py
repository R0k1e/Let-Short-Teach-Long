from pipeline import *

def tfidfTest(model, data_path):
    llm = OpenAI()
    llms = queue.Queue()
    llms.put(llm)
    paths = constructPath(data_path)
    with open(data_path, 'r') as file:
        for i, line in enumerate(file):
            obj = json.loads(line)
            text = obj['text']       
            pipeline_Level_tfidf(i+1, model, llms, text, paths)
    

def pipeline_Level_tfidf(data_id, model, llms, text, paths):
    llm = llms.get()
    try:
        lang = Generator_utils.identify_language(text.split('\n')[0])
        generator = Generatorllm(model, llm, lang)
        sumTree = SumTree_TFIDF(text, generator, lang = lang, chunk_size = 2048)
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
            questionMeta = generator.ask(node.getSummarywithImportance())
            question =questionMeta['question'] 
            answer, answerList = generator.pure_refine(sumTree.getTextArray(), question)
            Generator_utils.dumpIntermediate(data_id, paths["mapPath"], question, answerList, node)
            Generator_utils.dump(data_id, generator, paths["dataPath"], sumTree.getText(), questionMeta, answer, node)
    except Exception as e:
        raise e
    finally:
        llms.put(llm)



if __name__ == "__main__":
    model = "claude-3-sonnet-20240229"
    data_path = "experiment/UltraLink.json"
    tfidfTest(model, data_path)