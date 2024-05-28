from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
from langchain_text_splitters import TokenTextSplitter
import jieba
import json
 

text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",chunk_size=1024, chunk_overlap=0)
# Load a JSON file
with open('clean_data/threebody.jsonl', 'r') as f:
    data = json.load(f)
    text = data['text']
sents = text_splitter.split_text(text)
#首先，构建语料库corpus
sents=[jieba.lcut(sent) for sent in sents] #对每个句子进行分词
# sents=[word_tokenize(sent) for sent in sents] #对每个句子进行分词
#print(sents)  #输出分词后的结果
corpus=TextCollection(sents)  #构建语料库
print(corpus)  #输出语料库
 
# #计算语料库中"one"的tf值
# tf=corpus.tf('one',corpus)    # 1/12
# print(tf)
 
# #计算语料库中"one"的idf值
# idf=corpus.idf('one')      #log(3/1)
# print(idf)
 
# #计算语料库中"one"的tf-idf值
# tf_idf=corpus.tf_idf('one',corpus)
# print(tf_idf)

# Get top k words with highest TF-IDF values
k = 20
topk_words = sorted(set(word for word in corpus if len(word) > 1), key=lambda word: corpus.tf_idf(word, corpus), reverse=True)[:k]
topk_words_tf = sorted(set(word for word in corpus if len(word) > 1), key=lambda word: corpus.tf(word, corpus), reverse=True)[:k]

# Print top k words and their TF-IDF values
for word in topk_words:
    print(word)
    print(f"TF-IDF: {corpus.tf_idf(word, corpus)}")
    print('-----------------')
for word in topk_words_tf:
    print(word)
    print(f"TF: {corpus.tf(word, corpus)}")
    print('-----------------')
# for word in corpus:
#     print(word)
#     print(f"TF:{corpus.tf(word,corpus)}")
#     print(f"IDF:{corpus.idf(word)}")
#     print(f"TF-IDF:{corpus.tf_idf(word,corpus)}")
#     print('-----------------')
