from openai import OpenAI
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize, sent_tokenize
from langchain_text_splitters import TokenTextSplitter
import jieba
import json



def referenceConstruction(text):

  sentenses = sent_tokenize(text)
  context = ''
  for i, sentense in enumerate(sentenses):
      sentense = sentense.replace("\n", "")
      sentenseTemp = str(i+1)+'. '+sentense + " "
      context += sentenseTemp
      print(sentenseTemp)
  print(context)
  prompt = f"You need to summarize the following text. A reference need to be appended to each sentense in your summary. Here is the text:\n{context}\nRemenber that each sentence needs a reference to show the summary is from which sentence of original text. For example, \"The weather is not bad.(reference: 1, 2)\" That means the sentense is summarised from the first and second sentence of the original text. Your summary:"
  print(prompt)
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": f"{prompt}"}
    ]
  )

  print(completion.choices[0].message.content)
  with open ('experiment/summary.txt', 'a') as f:
      f.write('\n-----------------\n')
      f.write(context+'\n')
      f.write(completion.choices[0].message.content)




if __name__ == "__main__":
  client = OpenAI(
    api_key='',
    base_url=''
  ) 

  text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo",chunk_size=2048, chunk_overlap=0)
  # Load a JSON file
  with open('experiment/UltraLink.json', 'r') as f:
    data = json.load(f)
    text = data['text']
    sents = text_splitter.split_text(text)
    text = sents[0]
    referenceConstruction(text)