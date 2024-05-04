**#Requirements**

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
  * Prompt needs to be confirmed
* @Function: `ask`, ask a question based on given text
  * @Para: `summary`, the text to be questioned
  * Prompt needs to be confirmed
* @Function: `answer`, answer the question based on the given context
  * @Para: `context`
  * @Para: `question`
  * Prompt needs to be confirmed

***#Generator_utils.py***

* @Function: `clear`, clear a designate file
  * @Para: `path`, path to the file
* @Function: `dump`, dump data in json format to a file
  * @Para: `path`, path to the file
  * @Para: `text`
  * @Para: `question`
  * @Para: `answer`

**#sumTree.py**

*@Class: SumTree*

​	Class `SumTree` is a tree with nodes containing summaries. A SumTree is constructed based on a long text.

* @Init, build a SumTree based on a given long text
  * @Para: `text`, a long text
  * @Para: `summariser`, a generator
  * @Para: `chunk_size`, the size you want the chunk to be. Default: 2K
  * @Para: `children_group_capacity`, the capacity you want the group to be. Default: 3
* @Method: `getText`, get the original long text
* @Method: ` levelOrderTraversal`, traverse the tree in level order
* @Method: `nodeGraph`, draw a diagram of the entire tree
* @Method: `info`, print necessary information of the entire tree
* @Method: `getNode`, obtain a designate node
  * @Para: `level`
  * @Para: `index`
* @Method: `getRoot`
* @Method: `getTextArray`

*@Class: SumTreeNode*

​	Class `SumTreeNode` is a node in a `SumTree`.

* @Init
* @Method: `getChildrenSummarisation`, get all summaries from children
* @Method: `getSourceText`, get source text of this node