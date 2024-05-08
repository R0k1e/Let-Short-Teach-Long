from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI
import random
from Generator import Generator
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

    # summarise the text
    def summarise(self, text) -> str:
        self.__summarisation = self.__summariser.summary(text)

    # return the summarisation
    def getSummarisation(self) -> str:
        return self.__summarisation
    
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
        print(f"Source text of node {self.__level} {self.__index}:")
        print(self.__sourceTextArr)
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
    def __init__(self, text: str, summariser, chunk_size=2*1024, children_group_capacity=3):
        # self.text = text # save memory -- for efficiency
        self.__summariser : Generator= summariser
        self.type=""
        self.__textArray=[]
        self.__chunk_size = chunk_size # define the granularity of the summarisation
        self.__childern_group_capacity = children_group_capacity # define the number of children to be grouped together
        print(f"chunk_size: {self.__chunk_size}\nchildren_group_capacity: {self.__childern_group_capacity}")
        self.__height=0 # define the height of the tree
        self.__root=self.__build(text)
        self.nodeGraph()
        print("=====Create a SumTree successfully!=====")
    
    # chunk the text
    def __chunk(self, text, overlap=0):
        if isinstance(self.__summariser.llm, ChatOpenAI):
            text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name=self.__summariser.llm.model_name,chunk_size=self.__chunk_size, chunk_overlap=overlap)
        else:
            text_splitter = TokenTextSplitter.from_huggingface_tokenizer(self.__summariser.tokenizer, chunk_size=self.__chunk_size, chunk_overlap=overlap) # TODO: encoding method
        texts = text_splitter.split_text(text)
        print("=====Chunk successfully!=====")
        print(f"=====Chunk amount: {len(texts)}=====")
        return texts
    
    # build the tree
    def __build(self, text):
        chunks = self.__chunk(text)
        self.__textArray=chunks
        # try:
        #     self.type = self.__summariser.identifyType(random.choice(chunks))
        # except Exception as e:
        #     print(e)
        self.type = "text"
        children=[] # list of children
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
            child = SumTreeNode(self.__summariser)
            child.setTree(self)
            child.summarise(chunk)
            child.getSourceTextArr().append(i)
            children.append(child)
        parents = self.__group(children)
        if len(children) != 1 :
            self.__height+=1
        while len(parents)>1:
            self.__height+=1
            parents = self.__group(parents)
        print("=====Build successfully!=====")
        return parents[0] # TODO: return the root node
    
    # group the children and form the parent node
    def __group(self, children: list):
        if len(children)==1: # root node
            return children
        
        parents = []
        for i in range(len(children)):
            if i%self.__childern_group_capacity==0:
                group = SumTreeNode(self.__summariser)
                group.setTree(self)
                group.setChildren(children[i:i+self.__childern_group_capacity])

                for child in group.getChildren():
                    for source in child.getSourceTextArr():
                        group.getSourceTextArr().append(source)
                    child.setParent(group)
                
                text=group.getChildrenSummarisation()
                group.summarise(text)
                parents.append(group)
        print("=====Group successfully!=====")
        return parents

    # get original text -- for efficiency
    def getText(self):
        text=""
        for t in self.__textArray:
            text+=t
        return text
    
    # level order traversal
    def levelOrderTraversal(self):
        queue = [self.__root]
        while queue:
            node : SumTreeNode = queue.pop(0)
            print(f"LEVEL{node.getLevel()} NODE{node.getIndex()}: ")
            print(node.getSummarisation())
            children = node.getChildren()
            for child in children:
                queue.append(child)
    
    def levelOrderForDump(self):
        queue = [self.__root]
        result = {"Height": self.__height, "Type": self.type, "Chunk size": self.__chunk_size, "Children group capacity": self.__childern_group_capacity}
        while queue:
            node : SumTreeNode = queue.pop(0)
            result[f"LEVEL{node.getLevel()} NODE{node.getIndex()}"] = node.getSummarisation()
            children = node.getChildren()
            for child in children:
                queue.append(child)
        
        return result
    
    # print the tree structure
    def nodeGraph(self):
        print("=====Node Graph=====")
        queue = [self.__root]
        level=0
        while queue:
            level_nodes = len(queue)
            for i in range(level_nodes):
                node: SumTreeNode = queue.pop(0)
                if node is self.__root: 
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
        print(f"Height: {self.__height}")
        print(f"Type: {self.type}")
        print(f"Chunk size: {self.__chunk_size}")
        print(f"Children group capacity: {self.__childern_group_capacity}")
        print(f"Text length: {len(self.getText())}")
        print(f"Text array length: {len(self.__textArray)}")
        print(f"Root summarisation: {self.__root.getSummarisation()}")
        self.nodeGraph()

    # obtain the node at the specific level and index
    def getNode(self, level, index) -> SumTreeNode:
        if level == 0:
            return self.__root
        queue = [self.__root]
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
        return self.__root
    
    def getTextArray(self):
        return self.__textArray
    
    def getSourceSplitText(self, nodes : list[SumTreeNode]) -> list[str]:
        textArr = []
        for node in nodes:
            arr = node.getSourceTextArr()
            for i in arr:
                if i not in textArr:
                    textArr.append(i)
        
        return [self.__textArray[i] for i in textArr]

    def getRandomNode(self) -> SumTreeNode:
        level = random.randint(0, self.__height)
        queue = [self.__root]
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
        print(f"Random node: LEVEL{level} NODE{index}")
        return self.getNode(level, index)