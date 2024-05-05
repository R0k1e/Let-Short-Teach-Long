***#Requirements***

```
pip install langchain-text-splitters tiktoken
pip install langchain-openai tiktoken chromadb langchain langchainhub
pip install vllm
```

***#Generator_vllm.py***

*@Class: Generator*

​	Class `Generator` initialises a vllm process which will be used later.

* @Init
  * @Para: `model` , the path to the model
  * @Para: `num_gpus` , the number of gpus

* @Function: `sum`, summarise designate text
  * @Para: `text`, the text to be summarised
  * @Return: summary : `str`
  * Prompt needs to be confirmed
* @Function: `ask`, ask a question based on given text
  * @Para: `summary`, the text to be questioned
  * @Return: question : `str`
  * Prompt needs to be confirmed
* @Function: `answer`, answer the question based on the given context
  * @Para: `context`
  * @Para: `question`
  * @Return: answer : `str`
  * Prompt needs to be confirmed
* @Function: `mapReduce`, answer the question based on the given context in mapReduce way
  * @Para: `context`
  * @Para: `question`
  * @Return: answer : `str`
* @Function: `refine`, answer the question based on the given context in refine way
  * @Para: `context`
  * @Para: `question`
  * @Return: answer : `str`, now it contains intermediate data
* @Function: `mapRerank`, answer the question based on the given context in mapRerank way
  * @Para: `context`
  * @Para: `question`
  * @Return: answer : `str`, now it contains intermediate data

***#Generator_utils.py***

* @Function: `clear`, clear a designate file
  * @Para: `path`, path to the file
* @Function: `dump`, dump data in json format to a file
  * @Para: `path`, path to the file
  * @Para: `text`
  * @Para: `question`
  * @Para: `answer`
* @Function: `docParser`, transform a str list into a Document list
  * @Para: `docs`, the list to be transformed
  * @Return: a Document list : `list[Document]`

***#sumTree.py***

*@Class: SumTree*

​	Class `SumTree` is a tree with nodes containing summaries. A SumTree is constructed based on a long text.

* @Init, build a SumTree based on a given long text
  * @Para: `text`, a long text
  * @Para: `summariser`, a generator
  * @Para: `chunk_size`, the size you want the chunk to be. Default: 2K
  * @Para: `children_group_capacity`, the capacity you want the group to be. Default: 3
* @Function: `getText`, get the original long text
  * @Return: original text : `str`
* @Function: `levelOrderTraversal`, traverse the tree in level order
* @Function: `nodeGraph`, draw a diagram of the entire tree
* @Function: `info`, print necessary information of the entire tree
* @Function: `getNode`, obtain a designate node
  * @Para: `level`
  * @Para: `index`
  * @Return: node : `SumTreeNode`
* @Function: `getRoot`
  * @Return: root : `SumTreeNode`
* @Function: `getTextArray`
  * @Return: original text list : `list[str]`
* @Function: `getSourceSplitText`, get source text of nodes and return in list with *no repetitions*
  * @Para: `nodes` : `list[SumTreeNode]`, a list of nodes
  * @Return: source text list : `list[str]`

*@Class: SumTreeNode*

​	Class `SumTreeNode` is a node in a `SumTree`.

* @Init
* @Function: `getChildrenSummarisation`, get all summaries from children
  * @Return: summary : `str`
* @Function: `getSourceText`, get source text of this node
  * @Return: source text : `str`
* @Function: `getSourceSplitText`, get source text of this node in list
  * @Return: source text list : `list[str]`