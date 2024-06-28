from openai import OpenAI
import nltk
from nltk.text import TextCollection
from nltk.corpus import stopwords
import jieba
import json
import re
import Generator_utils
from Generatorllm import Generatorllm



def wordsSift(text, word_num = 400):
    stops = set(stopwords.words('english'))#blacklist
    if lang == 'zh':
        words = jieba.lcut(text)
    else :
        words = nltk.tokenize.word_tokenize(text)
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
    word_tokenize = nltk.tokenize.word_tokenize
    sent_tokenize = nltk.tokenize.sent_tokenize
    if lang == 'zh':
        word_tokenize = jieba.lcut
        sent_tokenize = Generator_utils.sent_tokenize_zh
    corpus=TextCollection([word_tokenize(chunk) for chunk in chunks])

    chunk_scores = {}
    for i, chunk in enumerate(chunks):
        sents = sent_tokenize(chunk)
        sent_scores = []
        for sent in sents:
            current_words = word_tokenize(sent)
            sent_score = 0
            for current_word in current_words:
                if current_word in siftedWords:
                    sent_score += corpus.tf_idf(current_word, corpus)
            sent_scores.append({"sent": sent, "score": sent_score})
        # sorted_scores = {k: v for k, v in sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)}
        chunk_scores[i] = sent_scores
    return chunk_scores


def referenceConstruction(text):
    sentenses = nltk.tokenize.sent_tokenize(text)
    context = ''
    for i, sentense in enumerate(sentenses):
        sentense = sentense.replace("\n", "")
        sentenseTemp = str(i+1)+'. '+sentense + " "
        context += sentenseTemp
  
    prompt = f"You need to create a summary of the provided text, ensuring each sentence in your summary references the specific sentence(s) from the original text it is derived from. Append a citation to each sentence in your summary to indicate its source. The output should be formatted in JSON, with each sentence of your summary as a key and the corresponding references as its value. For example, if the original text consists of three sentences describing the weather in London, your JSON response should map your summary sentences to their respective source sentences. Here's how to format your response:\nOriginal Text:\n1. Today, London is experiencing warm and dry weather, marking the peak of the current high temperatures.\n2. The day started with clear skies and temperatures around 14°C (57°F), gradually rising to a high of 24°C (75°F) in the afternoon.\n3. The weather is expected to remain sunny with only a few light clouds appearing later in the day. \nYour task:\nGenerate a JSON-formatted summary where each key is a summary sentence and each value is an array of sentence numbers from the original text that support the summary. \nExample response:\n{{\"It is a beautiful summer day in London.\": [1, 2, 3], \"The temperature may reach 24°C in the afternoon\": [2]}}\n Here is the text that you need to summarise:\n{context}\nYour summary:"
    references = generator.generate(prompt)
    
    return references


def referenceExtraction(text, sent_score):
    # text = "The authors present an open-source multilingual supervised fine-tuning dataset to improve the multilingual abilities of large language models (LLMs) (reference: 3). They introduce a knowledge-grounded data augmentation approach and demonstrate the strong cross-lingual transfer capabilities of modern LLMs, leading to substantial pruning of language-agnostic supervised fine-tuning data without performance degradation (reference: 5, 6, 7). The resulting UltraLink dataset comprises approximately 1 million samples across five languages and outperforms several representative baselines across various tasks (reference: 8, 9). The authors also propose a new approach to construct multilingual supervised fine-tuning data, emphasizing higher cultural diversity and pruned data volume (reference: 23, 24, 25, 28, 30). They engage in automatically generating multilingual supervised fine-tuning data and emphasize the importance of considering cultural diversity and integrating language-agnostic universal data (reference: 33, 34). The authors propose a data construction framework involving two pipelines and employ a knowledge-grounded data augmentation method based on Wikipedia dumps to improve cultural diversity (reference: 35, 37, 39, 40, 41). They use prompts to generate multi-turn dialogues conditioned on provided cultural backgrounds and dialogue history (reference: 49, 50, 51, 57, 58). The authors also propose two types of subsequent questions to improve the diversity of constructed dialogues (reference: 82, 83, 84)."

    result = text.replace("\n", "")
    references_ori = re.findall(r'\{.*?\}', result)
    references_ori = json.loads(references_ori[0])
    sent_scores = {}
    for sent, references in references_ori.items():
        sent_scores[sent] = 0
        for reference in references:
            sent_scores[sent] += sent_score[int(reference)-1]["score"]
    return sent_scores


# def referencePropagation(text): 
#     sent_references = referenceExtraction(text)
#     context = ''
#     for i, sent in enumerate(sent_references.keys()):
#         context += f"{i+1}. {sent} "
#     print(context)
#     prompt = f"You need to summarize the following text. A reference need to be appended to each sentense in your summary. Here is the text:\n{context}\nRemenber that each sentence needs a reference to show the summary is from which sentence of original text. For example, \"The weather is not bad.(reference: 1, 2)\" That means the sentense is summarised from the first and second sentence of the original text. Your summary:"
#     print(prompt)
    


if __name__ == "__main__":
    with open('experiment/UltraLink.json', 'r') as f:
        data = json.load(f)
        text = data['text']
    gpt = OpenAI()
    lang = Generator_utils.identify_language(text.split('\n')[0])
    generator = Generatorllm("gpt-3.5-turbo", gpt, lang)
    siftedWords = wordsSift(text)
    chunks = Generator_utils.splitTextOnSentences(text, generator.tokenizer, lang)
    for i, chunk in enumerate(chunks):
        print(i)
        print(chunk)
    chunk_score = sentenceScore(chunks, siftedWords)
    # print(chunk_score)
    for i, chunk in enumerate(chunks):
        references = referenceConstruction(chunk)
        sent_references = referenceExtraction(references, chunk_score[i])
        print(sent_references)

    
    # with open("experiment/summary.txt", "a") as f:
    #     f.write('\n-----------------\n')
    #     f.write(str(siftedWords)+'\n')

    # chunks = Generator_utils.splitTextOnSentences(text, generator.tokenizer, lang)
    # sentenceScores = sentenceScore(chunks, siftedWords)
    # print(referenceExtraction(""))