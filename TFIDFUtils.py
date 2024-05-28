from openai import OpenAI
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from langchain_text_splitters import TokenTextSplitter
import jieba
import json
import re
import tiktoken


def sent_tokenize_zh(text):
    resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    s = text
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def wordsSift(text):
    stops = set(stopwords.words('english'))#blacklist
    words = word_tokenize(text)
    words = set(words)
    words = [word for word in words if word not in stops]
    context = '['
    for word in words:
        context += word + ', '
    context = context[:-2] + ']'
    prompt = f"You need to sift the following words. You need to discard prepositions, conjunctions, articles and words that have no meanings. Here is the words:\n{context}\nRemenber that the output should be in json format. Here is an example: {{\"sifted_words\": [\"word1\", \"word2\", \"word3\"]}} Your sifted words:"
    # return prompt
    return words


def split_text_on_tokens(text: str, tokenizer, tokens_per_chunk, chunk_overlap = 0) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        if cur_idx == len(input_ids):
            break
        start_idx += tokens_per_chunk - chunk_overlap
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


def splitTextOnSentences(text, tokenizer, max_chunk_length=2048):
    sents = sent_tokenize(text)
    processed_sents = []
    for sent in sents:
        while len(tokenizer.encode(sent)) > max_chunk_length:
            texts = split_text_on_tokens(sent, tokenizer, max_chunk_length)
            processed_sents.append(texts[0])
            sent = ''.join(texts[1:])
        if sent:  # This ensures we add the last chunk if it's not empty
            processed_sents.append(sent)
    # Chunk sentences
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sent in processed_sents:
        if len(tokenizer.encode(current_chunk+sent)) >= max_chunk_length:
            chunks.append(current_chunk)
            current_chunk = ""
            current_length = 0
        current_chunk += sent + " "
        current_length += len(tokenizer.encode(sent))
    
    if current_chunk != "":
        chunks.append(current_chunk)
    
    return chunks


def sentenceScore(chunks, words):
    # chunks = splitTextOnSentences(text)
    corpus=TextCollection([word_tokenize(chunk) for chunk in chunks])

    chunk_scores = []
    for chunk in chunks:
        sents = sent_tokenize(chunk)
        sent_scores = {}
        for sent in sents:
            current_words = word_tokenize(sent)
            sent_score = 0
            for current_word in current_words:
                if current_word in words:
                    sent_score += corpus.tf_idf(current_word, corpus)
            sent_scores[sent] = sent_score
        sorted_scores = {k: v for k, v in sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)}
        chunk_scores.append(sorted_scores)
    return chunk_scores


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
    print(referenceExtraction(''))
    print(referencePropagation(''))