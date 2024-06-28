import random
from Generatorllm import Generatorllm
import Generator_utils
import TFIDFUtils
#from Summariser_vllm import Summariser


class SumTreeNode:
    def __init__(self, summariser):
        self.__summarisation = None
        self.__children = []
        self.__parent = None
        self.__summariser = summariser
        self.__level = 0
        self.__index = 0
        self.__sourceTextArr = []
        self.__tree= None
        self.sentScores = []


    # summarise the text
    def summarise(self, text) -> str:
        self.__summarisation = self.__summariser.summary(text, self.__tree.summarySize)
    

    def summariseWithRefine(self, previous, text) -> str:
        self.__summarisation = self.__summariser.summaryWithRefine(previous, text, self.__tree.summarySize)

    # return the summarisation
    def getSummarisation(self) -> str:
        return self.__summarisation
    
    def setSummarisation(self, summarisation):
        self.__summarisation = summarisation
    
    # obtain the children of the node
    def getChildren(self):
        return self.__children
    
    def setParent(self, parent):
        self.__parent = parent

    def getLevel(self):
        return self.__level
    
    def getIndex(self):
        return self.__index
    
    # obatin all the children summarisation
    def getChildrenSummarisation(self):
        text=""
        for child in self.__children:
            text+=child.getSummarisation()
        
        return text
    
    def getChildrenSentScores(self):
        scores = []
        for child in self.__children:
            scores+=child.sentScores
        return scores
    
    def getChildrenSize(self):
        return len(self.__children)
    
    def getSourceTextArr(self):
        return self.__sourceTextArr

    def setPosition(self, level, index):
        self.__level = level
        self.__index = index

    def getParent(self):
        return self.__parent
    
    def setChildren(self, children):
        self.__children = children

    def getSourceText(self):      
        text = ""
        for source in self.__sourceTextArr:
            text+=self.__tree.getTextArray()[source]
        return text
    
    def getSourceSplitText(self):
        texts = [self.__tree.getTextArray()[source] for source in self.__sourceTextArr]
        return texts
    
    def setTree(self, tree):
        self.__tree = tree

class SumTree:
    def __init__(self, text: str, summariser, chunk_size=2*1024, children_group_capacity=3, lang = 'en'):
        # self.text = text # save memory -- for efficiency
        self.summariser = summariser
        self.type=""
        self.textArray=[]
        self.summarySize = 0
        self.lang = lang
        self.chunk_size = chunk_size # define the granularity of the summarisation
        self.childern_group_capacity = children_group_capacity # define the number of children to be grouped together
        self.height=0 # define the height of the tree
        self.root=self.buildWithRefine(text) # now refine the summarisation
        self.nodeGraph()
        print("=====Create a SumTree successfully!=====")
    

    # chunk the text
    def chunk(self, text, overlap=0):
        # text_splitter = TokenTextSplitter.from_huggingface_tokenizer(self.__summariser.tokenizer, chunk_size=self.__chunk_size, chunk_overlap=overlap) 
        texts = Generator_utils.split_text_on_tokens(text, self.summariser.tokenizer, self.chunk_size, overlap)
            
        print(self.summariser.model_name)
        print("=====Chunk successfully!=====")
        print(f"=====Chunk amount: {len(texts)}=====")
        return texts
    
    # build the tree
    def build(self, text):
        chunks = self.chunk(text)
        self.textArray=chunks
        # try:
        #     self.type = self.__summariser.identifyType(random.choice(chunks))
        # except Exception as e:
        #     print(e)
        self.type = "text"
        children=[] # list of children
        for i, chunk in enumerate(chunks):
            if i == 0:
                self.summarySize = self.getSummarySize(chunk)
            child = SumTreeNode(self.summariser)
            child.setTree(self)
            child.summarise(chunk)
            child.getSourceTextArr().append(i)
            children.append(child)
        parents = self.group(children)
        if len(children) != 1 :
            self.height+=1
        while len(parents)>1:
            self.height+=1
            parents = self.group(parents)
        return parents[0] # TODO: return the root node
    

    def buildWithRefine(self, text):
        chunks = self.chunk(text)
        self.textArray=chunks
        # try:
        #     self.type = self.__summariser.identifyType(random.choice(chunks))
        # except Exception as e:
        #     print(e)
        self.type = "text"
        children=[] # list of children
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
            
            child = SumTreeNode(self.summariser)
            child.setTree(self)
            if i == 0:
                self.summarySize = self.getSummarySize(chunk)
                child.summarise(chunk)
            else: 
                child.summariseWithRefine(children[i-1].getSummarisation(), chunk)
            child.getSourceTextArr().append(i)
            children.append(child)
        parents = self.group(children)
        if len(children) != 1 :
            self.height+=1
        while len(parents)>1:
            self.height+=1
            parents = self.group(parents)
        print("=====Build successfully!=====")
        return parents[0] 
    

    # group the children and form the parent node
    def group(self, children: list):
        if len(children)==1: # root node
            return children
        
        parents = []
        for i in range(len(children)):
            if i%self.childern_group_capacity==0:
                group = SumTreeNode(self.summariser)
                group.setTree(self)
                group.setChildren(children[i:i+self.childern_group_capacity])

                for child in group.getChildren():
                    for source in child.getSourceTextArr():
                        group.getSourceTextArr().append(source)
                    child.setParent(group)
                
                text=group.getChildrenSummarisation()
                group.summarise(text)
                parents.append(group)
        return parents
    
    def getSummarySize(self, context: str):
        if self.lang == 'en':
            word_count = len(context.split())
            word_count = word_count//(self.childern_group_capacity+1)
            if word_count<300:
                word_count = 300
        elif self.lang == 'zh':
            word_count = len(context)
            word_count = word_count//(self.childern_group_capacity+1)
        else:
            word_count = 300 
        return word_count
        

    # get original text -- for efficiency
    def getText(self):
        text=""
        for t in self.textArray:
            text+=t
        return text
    
    # level order traversal
    def levelOrderTraversal(self) -> list[SumTreeNode]:
        nodes = []
        queue = [self.root]
        while queue:
            node : SumTreeNode = queue.pop(0)
            nodes.append(node)
            # print(f"LEVEL{node.getLevel()} NODE{node.getIndex()}: ")
            print(node.getSummarisation())
            children = node.getChildren()
            for child in children:
                queue.append(child)
        
        return nodes
    
    def levelOrderForDump(self):
        queue = [self.root]
        result = {"Height": self.height, "Type": self.type, "Chunk size": self.chunk_size, "Children group capacity": self.childern_group_capacity, "summarySize": self.summarySize}
        while queue:
            node : SumTreeNode = queue.pop(0)
            if node.getLevel() == self.height: 
                result[f"LEVEL{node.getLevel()} NODE{node.getIndex()} Input"] = node.getSourceText()
            else:
                result[f"LEVEL{node.getLevel()} NODE{node.getIndex()} Input"] = node.getChildrenSummarisation()
            result[f"LEVEL{node.getLevel()} NODE{node.getIndex()} Summary"] = node.getSummarisation()
            children = node.getChildren()
            for child in children:
                queue.append(child)
        
        return result
    
    # print the tree structure
    def nodeGraph(self):
        print("=====Node Graph=====")
        queue = [self.root]
        level=0
        while queue:
            level_nodes = len(queue)
            for i in range(level_nodes):
                node = queue.pop(0)
                if node is self.root: 
                    print(f"LEVEL0: ROOT", end=" ")
                    node.setPosition(0, 0)
                else:
                    if i==0:
                        print(f"LEVEL{level}: ", end="")
                    print(f"NODE{i}", end=" ")
                    node.setPosition(level, i)
                children = node.getChildren()
                for child in children:
                    queue.append(child)
            print()  # Print a new line after each level
            level+=1

    # print the information of the tree
    def info(self):
        print(f"Height: {self.height}")
        print(f"Type: {self.type}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Children group capacity: {self.childern_group_capacity}")
        print(f"Text length: {len(self.getText())}")
        print(f"Text array length: {len(self.textArray)}")
        print(f"Root summarisation: {self.root.getSummarisation()}")
        self.nodeGraph()

    # obtain the node at the specific level and index
    def getNode(self, level, index) -> SumTreeNode:
        if level == 0:
            return self.root
        queue = [self.root]
        current_level = 0
        while queue:
            if current_level == level:
                return queue[index]
            level_nodes = len(queue)
            for _ in range(level_nodes):
                node: SumTreeNode = queue.pop(0)
                children = node.getChildren()
                for child in children:
                    queue.append(child)
            current_level += 1
        return None

    def getRoot(self) -> SumTreeNode:
        return self.root
    
    def getTextArray(self):
        return self.textArray
    
    def getSourceSplitText(self, nodes : list[SumTreeNode]) -> list[str]:
        textArr = []
        for node in nodes:
            arr = node.getSourceTextArr()
            for i in arr:
                if i not in textArr:
                    textArr.append(i)
        
        return [self.textArray[i] for i in textArr]

    def getRandomNode(self) -> SumTreeNode:
        level = random.randint(0, self.height)
        queue = [self.root]
        levelNow = 0
        while queue:
            if levelNow == level:
                index = random.randint(0, len(queue)-1)
                break
            level_nodes = len(queue)
            for i in range(level_nodes):
                node: SumTreeNode = queue.pop(0)
                children = node.getChildren()
                for child in children:
                    queue.append(child)
            levelNow+=1
        return self.getNode(level, index)
    
    def getChildrenGroupCapacity(self):
        return self.childern_group_capacity
    
    def getNodeEachLevel(self) -> list[SumTreeNode]:
        queue = [self.root]
        nodes = []
        levelNow = 0
        while queue:
            index = random.randint(0, len(queue)-1)            
            level_nodes = len(queue)
            for i in range(level_nodes):
                node: SumTreeNode = queue.pop(0)
                children = node.getChildren()
                for child in children:
                    queue.append(child)
            nodes.append(self.getNode(levelNow, index))
            levelNow+=1
        return nodes
    

class SumTree_TFIDF(SumTree):
    def __init__(self, text: str, summariser, chunk_size=2*1024, children_group_capacity=3, lang = 'en'):
        super().__init__(text, summariser, chunk_size, children_group_capacity, lang)
    

    def buildWithRefine(self, text):
        chunks = self.chunk(text)
        self.textArray=chunks
        # try:
        #     self.type = self.__summariser.identifyType(random.choice(chunks))
        # except Exception as e:
        #     print(e)
        self.type = "text"
        siftedWords = self.summariser.wordsSift(text)
        sentScores = self.summariser.sentenceScore(chunks, siftedWords)
        children=[] # list of children
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
            
            child = SumTreeNode(self.summariser)
            child.setTree(self)
            summary = ""
            if i == 0:
                self.summarySize = self.getSummarySize(chunk)
                references = self.summariser.referenceConstruction(chunk)
                child.sentScores = self.summariser.referenceExtraction(references, sentScores[i])
                for item in child.sentScores:
                    summary += item["sent"]
                    summary += " "
                child.setSummarisation(summary)
            else: 
                references = self.summariser.referenceConstructionwithRefine(children[i-1].getSummarisation(), chunk)
                child.sentScores = self.summariser.referenceExtraction(references, sentScores[i])
                for item in child.sentScores:
                    summary += item["sent"]
                    summary += " "
                child.setSummarisation(summary)
            child.getSourceTextArr().append(i)
            children.append(child)
        parents = self.group(children)
        if len(children) != 1 :
            self.height+=1
        while len(parents)>1:
            self.height+=1
            parents = self.group(parents)
        print("=====Build successfully!=====")
        return parents[0]

    

    def group(self, children: list):
        if len(children)==1: # root node
            return children
        
        parents = []
        for i in range(len(children)):
            if i%self.childern_group_capacity==0:
                group = SumTreeNode(self.summariser)
                group.setTree(self)
                group.setChildren(children[i:i+self.childern_group_capacity])

                for child in group.getChildren():
                    for source in child.getSourceTextArr():
                        group.getSourceTextArr().append(source)
                    child.setParent(group)
                
                text=group.getChildrenSummarisation()
                references = self.summariser.referenceConstruction(text)
                summary = ""
                group.sentScores = self.summariser.referenceExtraction(references, group.getChildrenSentScores())
                for item in group.sentScores:
                    summary += item["sent"]
                    summary += " "
                group.setSummarisation(summary)
                parents.append(group)
        return parents