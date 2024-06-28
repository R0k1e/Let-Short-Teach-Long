from openai import OpenAI
from transformers import GPT2TokenizerFast, AutoTokenizer
from nltk.text import TextCollection
from nltk.corpus import stopwords
import Generator_utils
import random
import time
import tiktoken
import nltk
import jieba
import json
import re

maximum_tokens = 4*1024

# class GenerateFailedException(Exception):
#     def __init__(self, task):
#         self.task =task

    
#     def __str__(self):
#         return f"Failed to finish {self.task}"
    

class Generatorllm:
    def __init__(
            self,
            model,
            client,
            lang = 'en'
        ):

        self.model = model
        self.model_name = model.split("/")[-1]
        if "gpt" in self.model_name:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        elif "claude" in self.model_name:
            self.tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lang = lang
        self.client = client


    def formPrompt(self, task):
        modelTemplate = Generator_utils.getTemplate(self.model_name)
        prompt = modelTemplate.format(user_message = Generator_utils.getPrompt(task, self.lang))
        return prompt
    

    def formCompletion(self, prompt):
        length = self.checkLength(prompt)
        while self.checkLength(prompt) > maximum_tokens-20:
            prompt = prompt[:-10]
        length = self.checkLength(prompt)
        # result = self.client.completions.create(model = self.model, prompt = prompt, max_tokens = maximum_tokens-length)
        # result = result.choices[0].text
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages= [
                {
                    "role": "user",
                    "content": prompt,
                },
            ], 
            max_tokens=maximum_tokens-length
        )
        result = completion.choices[0].message.content
        
        with open("debugOutput.txt", "a") as f:
            f.write("##PROMPT##\n")
            f.write(prompt+"\n")
            f.write("##LENGTH##\n")
            f.write(str(self.checkLength(prompt))+"\n")
            f.write("##RESPONSE##\n")
            f.write(result+"\n")
            f.write("##LENGTH##\n")
            f.write(str(self.checkLength(result))+"\n")
        return result
    

    def checkLength(self, text):
        length = len(self.tokenizer.encode(text))
        # breakpoint()
        # if length > 4096:
        #     raise GenerateFailedException("checkLength")
        return length
    

    def generate(self, text: str):
        prompt = self.formPrompt("generate")
        prompt = prompt.format(text=text)
        
        result = self.formCompletion(prompt)

        return result


    def summary(self, text: str, word_count):
        prompt = self.formPrompt("summary")
        prompt = prompt.format(context=text, word_count=word_count)
        result = self.formCompletion(prompt)
        return result
    

    def summaryWithRefine(self, previous: str, text: str, word_count):
        prompt = self.formPrompt("summaryWithRefine")
        prompt = prompt.format(previousSummary=previous, context=text, word_count=word_count)
        result = self.formCompletion(prompt)
        return result


    def ask(self, context) -> dict[str]:
        random.seed(time.time())
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategoryKey = random.choice(list(questionCategories.keys()))
        questionCategory = questionCategories[questionCategoryKey]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehensionKey = random.choice(list(comprehensions.keys()))
        comprehension = comprehensions[comprehensionKey]
        prompt = self.formPrompt("ask")
        prompt = prompt.format(questionCategory=questionCategory, comprehension=comprehension, context=context)
        question = self.formCompletion(prompt)
        question = question.strip()
        return {"question":question, "questionCategory": questionCategoryKey, "comprehension":comprehensionKey}


    def askwithMeta(self, context, questionMeta: dict) -> dict[str]:
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategory = questionCategories[questionMeta["questionCategory"]]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehension = comprehensions[questionMeta["comprehension"]]
        prompt = self.formPrompt("ask")
        prompt = prompt.format(questionCategory=questionCategory, comprehension=comprehension, context=context)
        question = self.formCompletion(prompt)
        question = question.strip()
        return {"question":question, "questionCategory": questionMeta["questionCategory"], "comprehension":questionMeta["comprehension"]}
    

    def ask_all_type(self, context, questionMeta: dict) -> dict[str]:
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategory = questionCategories[questionMeta["questionCategory"]]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehension = comprehensions[questionMeta["comprehension"]]
        prompt = self.formPrompt("ask")
        prompt = prompt.format(questionCategory=questionCategory, comprehension=comprehension, context=context)
        question = self.formCompletion(prompt)
        question = question.strip()
        return {"question":question, "questionCategory": questionMeta["questionCategory"], "comprehension":questionMeta["comprehension"]}
    
    
    def identifyType(self, context):
        prompt = self.formPrompt("identifyType")
        prompt = prompt.format(context=context)
        result = self.formCompletion(prompt)
        print(result)
        match = re.search(r'Type:\s*(.+)', result)
        if match:
            textType_name = match.group(1).strip(" .\n")
            print(textType_name)
            return textType_name
        else:
            print("textType Name not found.")
            raise Exception("textType Name not found.")


    def mr_map(self, context: list[str], question):      
        answer = ""
        answerList = {}
        print("=====Map=====")
        for i, item in enumerate(context):
            map_prompt = self.formPrompt("extract")  
            map_prompt = map_prompt.format(context=item, question=question)
            result = self.formCompletion(map_prompt)
            # print(f"Chunk {i+1}/{len(context)}:\n{result}")
            answer += result
            answerList[f"Chunk {i+1}"] = result
        return answer, answerList


    def mr_reduce(self, context, question):
        # Reduce
        reduce_prompt = self.formPrompt("firstAnswer")
        reduce_prompt = reduce_prompt.format(context=context, question=question)
        result = self.formCompletion(reduce_prompt)
        return result


    def pure_refine(self, context: list[str], question):
        result = ""
        answerList = {"generator": self.model_name}
        for i in range(len(context)):
            if i == 0:
                prompt = self.formPrompt("firstAnswer")
                prompt = prompt.format(context=context[i], question=question)
                result = self.formCompletion(prompt)
            else:
                prompt = self.formPrompt("followingAnswer")
                prompt = prompt.format(context=context[i], answer=result, question=question)
                result = self.formCompletion(prompt)
            answerList[f"Intermediate input {i}"] = context[i]
            answerList[f"Intermediate result {i}"] = result
        return result, answerList
    

    #对多个chunk进行信息提取，然后对提取的信息进行拼接
    def refine(self, context: list[str], question, group_size=3):
        intermediate = ""
        result = ""
        answerList = {"generator": self.model_name}
        for i in range(len(context)):
            prompt = self.formPrompt("extract")
            prompt = prompt.format(context=context[i], question=question)
            extract_result = self.formCompletion(prompt)
            intermediate += extract_result
            answerList["Chunk "+str(i+1)] = context[i]
            answerList["Extracted "+str(i+1)] = extract_result
            if i % group_size == group_size - 1 or i == len(context) - 1:
                if i // group_size == 0:
                    prompt = self.formPrompt("firstAnswer")
                    prompt = prompt.format(context=intermediate, question=question)
                    result = self.formCompletion(prompt)
                else:
                    prompt = self.formPrompt("followingAnswer")
                    prompt = prompt.format(context=intermediate, answer=result, question=question)
                    result = self.formCompletion(prompt)
                answerList["Intermediate input"+str(i//group_size)] = intermediate
                answerList["Intermediate result"+str(i//group_size)] = result
                intermediate = ""
        return result, answerList
    

    def extract(self, context, question):
        prompt = self.formPrompt("extract")
        prompt = prompt.format(context=context, question=question)
        extract_result = self.formCompletion(prompt)
        return extract_result
    

    def firstAnswer(self, context, question):
        prompt = self.formPrompt("firstAnswer")
        prompt = prompt.format(context=context, question=question)
        result = self.formCompletion(prompt)
        return result


    def followingAnswer(self, context, question, previousAnswer):
        prompt = self.formPrompt("followingAnswer")
        prompt = prompt.format(context=context, answer=previousAnswer, question=question)
        result = self.formCompletion(prompt)
        return result
    

    #TF_IDF components
    def wordsSift(self, text, word_num = 400):
        stops = set(stopwords.words('english'))#blacklist
        if self.lang == 'zh':
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
            prompt = self.formPrompt("wordsSift")
            prompt = prompt.format(context = context)
            result = self.formCompletion(prompt)
            result = result.replace("\n", "")
            siftedWords = re.findall(r'\{.*?\}', result)
            tempWords = json.loads(siftedWords[0])["sifted_words"]
            finalWords += tempWords
        finalWords = list(set(finalWords))
        return finalWords

    # decreasing order
    def sentenceScore(self, chunks, siftedWords):
        # chunks = splitTextOnSentences(text)
        word_tokenize = nltk.tokenize.word_tokenize
        sent_tokenize = nltk.tokenize.sent_tokenize
        if self.lang == 'zh':
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


    def referenceConstruction(self, text):
        if self.lang == 'zh':
            sentenses = Generator_utils.sent_tokenize_zh(text)
        else:
            sentenses = nltk.tokenize.sent_tokenize(text)
        context = ''
        for i, sentense in enumerate(sentenses):
            sentense = sentense.replace("\n", "")
            sentenseTemp = str(i+1)+'. '+sentense + " "
            context += sentenseTemp
    
        prompt = self.formPrompt("referenceConstruction")
        prompt = prompt.format(context = context)
        references = self.formCompletion(prompt)
        
        return references
    

    def referenceConstructionwithRefine(self, previous: str, text: str):
        if self.lang == 'zh':
            sentenses = Generator_utils.sent_tokenize_zh(text)
        else:
            sentenses = nltk.tokenize.sent_tokenize(text)
        context = ''
        for i, sentense in enumerate(sentenses):
            sentense = sentense.replace("\n", "")
            sentenseTemp = str(i+1)+'. '+sentense + " "
            context += sentenseTemp
    
        prompt = self.formPrompt("referenceConstructionwithRefine")
        prompt = prompt.format(previousSummary = previous, context = context)
        references = self.formCompletion(prompt)
        
        return references


    def referenceExtraction(self, text, sent_score):
        # text = "The authors present an open-source multilingual supervised fine-tuning dataset to improve the multilingual abilities of large language models (LLMs) (reference: 3). They introduce a knowledge-grounded data augmentation approach and demonstrate the strong cross-lingual transfer capabilities of modern LLMs, leading to substantial pruning of language-agnostic supervised fine-tuning data without performance degradation (reference: 5, 6, 7). The resulting UltraLink dataset comprises approximately 1 million samples across five languages and outperforms several representative baselines across various tasks (reference: 8, 9). The authors also propose a new approach to construct multilingual supervised fine-tuning data, emphasizing higher cultural diversity and pruned data volume (reference: 23, 24, 25, 28, 30). They engage in automatically generating multilingual supervised fine-tuning data and emphasize the importance of considering cultural diversity and integrating language-agnostic universal data (reference: 33, 34). The authors propose a data construction framework involving two pipelines and employ a knowledge-grounded data augmentation method based on Wikipedia dumps to improve cultural diversity (reference: 35, 37, 39, 40, 41). They use prompts to generate multi-turn dialogues conditioned on provided cultural backgrounds and dialogue history (reference: 49, 50, 51, 57, 58). The authors also propose two types of subsequent questions to improve the diversity of constructed dialogues (reference: 82, 83, 84)."

        result = text.replace("\n", "")
        references_ori = re.findall(r'\{.*?\}', result)
        references_ori = json.loads(references_ori[0])
        sent_scores = []
        for sent, references in references_ori.items():
            temp_score = {"sent": sent, "score": 0}
            for reference in references:
                temp_score["score"] += sent_score[int(reference)-1]["score"]
            sent_scores.append(temp_score)
        return sent_scores
    


        
