from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json

# tokenizer = AutoTokenizer.from_pretrained("./dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("./dslim/bert-base-NER")

# nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# # text = "My name is wolfgang and I live in berlin"
# with open('experiment/UltraLink.json', 'r') as f:
#         data = json.load(f)
#         text = data['text']

# ner_results = nlp(text)
# print(ner_results)



tokenizer = AutoTokenizer.from_pretrained("./Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("./Jean-Baptiste/roberta-large-ner-english")

with open('experiment/UltraLink.json', 'r') as f:
        data = json.load(f)
        text = data['text']

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = nlp(text)
ENs = []
for result in results:
    entity = result['word'].strip()
    ENs.append(entity)

ENs = list(set(ENs))
print(ENs)