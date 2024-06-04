from openai import OpenAI
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
import Generator_utils
import re
import random
import time
import tiktoken
import os

maximum_tokens = 128*1024

class GenerateFailedException(Exception):
    def __init__(self, task):
        self.task =task

    
    def __str__(self):
        return f"Failed to finish {self.task}"
    

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
            f.write("##RESPONSE##\n")
            f.write(result+"\n")
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
        try:
            result = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("generate")
        return result


    def summary(self, text: str, word_count):
        prompt = self.formPrompt("summary")
        prompt = prompt.format(context=text, word_count=word_count)
        try: 
            result = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("summary")
        return result
    

    def summaryWithRefine(self, previous: str, text: str, word_count):
        prompt = self.formPrompt("summaryWithRefine")
        prompt = prompt.format(previousSummary=previous, context=text, word_count=word_count)
        try:
            result = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("summaryWithRefine")
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
        try:
            question = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        question = question.strip()
        return {"question":question, "questionCategory": questionCategoryKey, "comprehension":comprehensionKey}


    def askwithMeta(self, context, questionMeta: dict) -> dict[str]:
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategory = questionCategories[questionMeta["questionCategory"]]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehension = comprehensions[questionMeta["comprehension"]]
        prompt = self.formPrompt("ask")
        prompt = prompt.format(questionCategory=questionCategory, comprehension=comprehension, context=context)
        try:
            question = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        question = question.strip()
        return {"question":question, "questionCategory": questionMeta["questionCategory"], "comprehension":questionMeta["comprehension"]}
    

    def ask_all_type(self, context, questionMeta: dict) -> dict[str]:
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategory = questionCategories[questionMeta["questionCategory"]]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehension = comprehensions[questionMeta["comprehension"]]
        prompt = self.formPrompt("ask")
        prompt = prompt.format(questionCategory=questionCategory, comprehension=comprehension, context=context)
        try:
            question = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        question = question.strip()
        return {"question":question, "questionCategory": questionMeta["questionCategory"], "comprehension":questionMeta["comprehension"]}
    
    
    def identifyType(self, context):
        prompt = self.formPrompt("identifyType")
        prompt = prompt.format(context=context)
        try:
            result = self.formCompletion(prompt)
        except Exception as e:
            print(e)
            raise GenerateFailedException("identifyType")
        print(result)
        match = re.search(r'Type:\s*(.+)', result)
        if match:
            textType_name = match.group(1).strip(" .\n")
            print(textType_name)
            return textType_name
        else:
            print("textType Name not found.")
            raise GenerateFailedException("identifyType")


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