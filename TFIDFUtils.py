from langchain_openai import ChatOpenAI
from openai import OpenAI
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import jieba
import json
import re
import tiktoken
import Generator_utils
from Generatorllm import Generatorllm



def wordsSift(text, word_num = 512):
    stops = set(stopwords.words('english'))#blacklist
    if lang == 'zh':
        word_tokenize = jieba.lcut
    words = word_tokenize(text)
    words = set(words)
    words = [word for word in words if word not in stops]
    finalWords = []
    for i in range(len(words)//word_num):
        tempWords = words[i*word_num:(i+1)*word_num]
        context = '['
        for word in tempWords:
            context += word + ', '
        context = context[:-2] + ']'
        prompt = f"You need to sift the following words. You need to discard prepositions, conjunctions, articles and words that have no meanings. Here is the words:\n{context}\nRemenber that the output should be in json format. Here is an example: {{\"sifted_words\": [\"word1\", \"word2\", \"word3\"]}} Your sifted words:"
        result = generator.generate(prompt)
        # client = OpenAI(base_url="")
        # result = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": f"{prompt}"}
        #     ]
        # )
        # result = result.choices[0].message.content
        print(result)
        result = result.replace("\n", "")
        siftedWords = re.findall(r'\{.*?\}', result)
        tempWords = json.loads(siftedWords[0])["sifted_words"]
        finalWords += tempWords
    finalWords = list(set(finalWords))
    return finalWords

# decreasing order
def sentenceScore(chunks, siftedWords):
    # chunks = splitTextOnSentences(text)
    if lang == 'zh':
        word_tokenize = jieba.lcut
        sent_tokenize = Generator_utils.sent_tokenize_zh
    corpus=TextCollection([word_tokenize(chunk) for chunk in chunks])

    chunk_scores = {}
    for i, chunk in enumerate(chunks):
        sents = sent_tokenize(chunk)
        sent_scores = {}
        for sent in sents:
            current_words = word_tokenize(sent)
            sent_score = 0
            for current_word in current_words:
                if current_word in siftedWords:
                    sent_score += corpus.tf_idf(current_word, corpus)
            sent_scores[sent] = sent_score
        sorted_scores = {k: v for k, v in sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)}
        chunk_scores[i] = sorted_scores
    return chunk_scores


# def globalNumbering(texts):
#     text = ""
#     for tempText in texts:
#         text += tempText
#     if lang == 'zh':
#         sent_tokenize = Generator_utils.sent_tokenize_zh
#     sents = sent_tokenize(text)
#     numbering = {}
#     for i, sent in enumerate(sents):
#         numbering[sent] = i
#     return numbering


# def markImportant(text, scores):


def referenceConstruction(text):
    sentenses = sent_tokenize(text)
    context = ''
    for i, sentense in enumerate(sentenses):
        sentense = sentense.replace("\n", "")
        sentenseTemp = str(i+1)+'. '+sentense + " "
        context += sentenseTemp
  
    prompt = f"You need to summarize the following text. A reference need to be appended to each sentense in your summary. Here is the text:\n{context}\nRemenber that each sentence needs a reference to show the summary is from which sentence of original text. For example, \"The weather is not bad.(reference: 1, 2)\" That means the sentense is summarised from the first and second sentence of the original text. Your summary:"
    print(prompt)
    # completion = client.chat.completions.create(
    #   model="gpt-3.5-turbo",
    #   messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": f"{prompt}"}
    #   ]
    # )

    # print(completion.choices[0].message.content)
    # with open ('experiment/summary.txt', 'a') as f:
    #     f.write('\n-----------------\n')
    #     f.write(context+'\n')
    #     f.write(completion.choices[0].message.content)


def referenceExtraction(text):
    text = "The authors present an open-source multilingual supervised fine-tuning dataset to improve the multilingual abilities of large language models (LLMs) (reference: 3). They introduce a knowledge-grounded data augmentation approach and demonstrate the strong cross-lingual transfer capabilities of modern LLMs, leading to substantial pruning of language-agnostic supervised fine-tuning data without performance degradation (reference: 5, 6, 7). The resulting UltraLink dataset comprises approximately 1 million samples across five languages and outperforms several representative baselines across various tasks (reference: 8, 9). The authors also propose a new approach to construct multilingual supervised fine-tuning data, emphasizing higher cultural diversity and pruned data volume (reference: 23, 24, 25, 28, 30). They engage in automatically generating multilingual supervised fine-tuning data and emphasize the importance of considering cultural diversity and integrating language-agnostic universal data (reference: 33, 34). The authors propose a data construction framework involving two pipelines and employ a knowledge-grounded data augmentation method based on Wikipedia dumps to improve cultural diversity (reference: 35, 37, 39, 40, 41). They use prompts to generate multi-turn dialogues conditioned on provided cultural backgrounds and dialogue history (reference: 49, 50, 51, 57, 58). The authors also propose two types of subsequent questions to improve the diversity of constructed dialogues (reference: 82, 83, 84)."

    positions = re.finditer(r'\(reference: [\d,\s]+\)', text)
    sents = []
    start = 0
    for position in positions:
        sent = text[start:position.start()]
        sent = sent.strip(". ")
        sent += "."
        sents.append(sent)
        start = position.end()
    references = re.findall(r'\(reference: ([\d, ]+)\)', text)
    sent_references = dict(zip(sents, references))
    return sent_references



def referencePropagation(text): 
    sent_references = referenceExtraction(text)
    context = ''
    for i, sent in enumerate(sent_references.keys()):
        context += f"{i+1}. {sent} "
    print(context)
    prompt = f"You need to summarize the following text. A reference need to be appended to each sentense in your summary. Here is the text:\n{context}\nRemenber that each sentence needs a reference to show the summary is from which sentence of original text. For example, \"The weather is not bad.(reference: 1, 2)\" That means the sentense is summarised from the first and second sentence of the original text. Your summary:"
    print(prompt)
    


if __name__ == "__main__":
    # client = OpenAI(
    #   api_key='',
    #   base_url=''
    # ) 

    # text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",chunk_size=2048, chunk_overlap=0)
    # # Load a JSON file
    # with open('experiment/UltraLink.json', 'r') as f:
    #     data = json.load(f)
    #     text = data['text']
        # sents = text_splitter.split_text(text)
        # text = sents[0]
        # referenceConstruction(text)
        # print(wordsSift(text))
        # results = sentenceScore(text, wordsSift(text))
        # for result in results:
        #     print("Chunk")
        #     for sent, score in result.items():
        #         print(sent)
        #         print(score)
        #         print('-----------------')
    # stopWords()
    # print(referenceExtraction(''))
    # print(referencePropagation(''))
    with open('experiment/UltraLink.json', 'r') as f:
        data = json.load(f)
        text = data['text']
    gpt = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    lang = Generator_utils.identify_language(text.split('\n')[0])
    generator = Generatorllm(gpt, encoding, lang)
    siftedWords = wordsSift(text)
    
    with open("experiment/summary.txt", "a") as f:
        f.write('\n-----------------\n')
        f.write(str(siftedWords)+'\n')

    chunks = Generator_utils.splitTextOnSentences(text, generator.tokenizer, lang)
    sentenceScores = sentenceScore(chunks, siftedWords)
