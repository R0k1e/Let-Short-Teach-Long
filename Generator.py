import time
from langchain_community.llms import VLLM, VLLMOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import Generator_utils
from langchain_core.output_parsers import StrOutputParser
from GenerateFailedException import GenerateFailedException
from transformers import AutoTokenizer
import Generator_utils
import os, re
import random


class Generator:
    def __init__(
            self,
            llm,
            tokenizer = None,
            lang = 'en'
        ):

        self.llm = llm
        self.tokenizer = tokenizer
        self.lang = lang
        if isinstance(self.llm, ChatOpenAI):
            self.model_name = self.llm.model_name
        elif isinstance(self.llm, VLLMOpenAI):
            self.model_name = self.llm.model_name.split("/")[-1]
        else:
            self.model_name = self.llm.model.split("/")[-1]
        print("=====Create a summariser successfully!=====")


    def _formPrompt(self, task):
        modelTemplate = Generator_utils.getTemplate(self.model_name)
        modelTemplate = modelTemplate.format(user_message= Generator_utils.getPrompt(task, self.lang))
        prompt = PromptTemplate.from_template(modelTemplate)
        return prompt
    
    def _formChain(self, prompt):
        llm_chain = None
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            llm_chain = prompt | self.llm | outputParser
        else:
            llm_chain = prompt | self.llm 

        return llm_chain

    def summary(self, text: str, word_count):
        prompt = self._formPrompt("summary")
        llm_chain = self._formChain(prompt)
        
        
        try: 
            summary_result = llm_chain.invoke({"context": text, "word_count": word_count})
        except Exception as e:
            print(e)
            raise GenerateFailedException("summary")
        print(summary_result)
        print("=====Summarise successfully!=====")
        return summary_result

    def ask(self, context) -> dict[str]:
        random.seed(time.time())
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategoryKey = random.choice(list(questionCategories.keys()))
        questionCategory = questionCategories[questionCategoryKey]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehensionKey = random.choice(list(comprehensions.keys()))
        comprehension = comprehensions[comprehensionKey]
        prompt = self._formPrompt("ask")
        llm_chain = self._formChain(prompt)

        try:
            question = llm_chain.invoke({"questionCategory": questionCategory, "comprehension": comprehension, "context": context})
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        question = question.strip()
        print(question)
        print("=====Ask successfully!=====")
        return {"question":question, "questionCategory": questionCategoryKey, "comprehension":comprehensionKey}

    def ask_all_type(self, context, questionMeta: dict) -> dict[str]:
        questionCategories = Generator_utils.getPrompt("questionCategory", self.lang)
        questionCategory = questionCategories[questionMeta["questionCategory"]]
        comprehensions = Generator_utils.getPrompt("comprehension", self.lang)
        comprehension = comprehensions[questionMeta["comprehension"]]
        prompt = self._formPrompt("ask")
        llm_chain = self._formChain(prompt)

        try:
            question = llm_chain.invoke({"questionCategory": questionCategory, "comprehension": comprehension, "context": context})
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        question = question.strip()
        print(question)
        print("=====Ask successfully!=====")
        return {"question":question, "questionCategory": questionMeta["questionCategory"], "comprehension":questionMeta["comprehension"]}
    
    
    def identifyType(self, context):
        prompt = self._formPrompt("identifyType")
        
        llm_chain = self._formChain(prompt)

        try:
            result = llm_chain.invoke({"context": context})
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
        map_prompt = self._formPrompt("extract")        
        map_chain = self._formChain(map_prompt)
        answer = ""
        answerList = {}
        print("=====Map=====")
        for i, item in enumerate(context):
            result = map_chain.invoke({"question": question, "context": item})
            print(f"Chunk {i+1}/{len(context)}:\n{result}")
            answer += result
            answerList[f"Chunk {i+1}"] = result
        
        return answer, answerList
        
    def mr_reduce(self, context, question):
        # Reduce
        reduce_prompt = self._formPrompt("firstAnswer")
        
        reduce_chain = self._formChain(reduce_prompt)
        print("=====Reduce=====")
        result = reduce_chain.invoke({"context": context, "question": question})
        print(result)
        return result

    def pure_refine(self, context: list[str], question):
        result = ""
        answerList = {"generator": self.model_name}
        for i in range(len(context)):
                if i == 0:
                    prompt = self._formPrompt("firstAnswer")
                    chain = self._formChain(prompt)
                    result = chain.invoke({"context": context[i], "question": question})
                else:
                    prompt = self._formPrompt("followingAnswer")
                    chain = self._formChain(prompt)
                    result = chain.invoke({"context": context[i], "answer": result, "question": question})
                answerList[f"Intermediate input {i}"] = context[i]
                answerList[f"Intermediate result {i}"] = result
                print(f"Intermediate result:\n{result}")
        return result, answerList
    

    #对多个chunk进行信息提取，然后对提取的信息进行拼接
    def refine(self, context: list[str], question, group_size=3):
        intermediate = ""
        result = ""
        answerList = {"generator": self.model_name}
        for i in range(len(context)):
            prompt = self._formPrompt("extract")
            chain = self._formChain(prompt)
            extract_result = chain.invoke({"context": context[i], "question": question})
            print(f"Chunk {i+1}/{len(context)}:\n{extract_result}")
            intermediate += extract_result
            answerList["Chunk "+str(i+1)] = context[i]
            answerList["Extracted "+str(i+1)] = extract_result
            if i % group_size == group_size - 1 or i == len(context) - 1:
                if i // group_size == 0:
                    prompt = self._formPrompt("firstAnswer")
                    chain = self._formChain(prompt)
                    result = chain.invoke({"context": intermediate, "question": question})
                else:
                    prompt = self._formPrompt("followingAnswer")
                    chain = self._formChain(prompt)
                    result = chain.invoke({"context": intermediate, "answer": result, "question": question})
                answerList["Intermediate input"+str(i//group_size)] = intermediate
                answerList["Intermediate result"+str(i//group_size)] = result
                print(f"Intermediate result:\n{result}")
                intermediate = ""
        return result, answerList
    
    def extract(self, context, question):
        prompt = self._formPrompt("extract")
        chain = self._formChain(prompt)
        extract_result = chain.invoke({"context": context, "question": question})
        return extract_result
    
    def firstAnswer(self, context, question):
        prompt = self._formPrompt("firstAnswer")
        chain = self._formChain(prompt)
        result = chain.invoke({"context": context, "question": question})
        return result
        
    def followingAnswer(self, context, question, previousAnswer):
        prompt = self._formPrompt("followingAnswer")
        chain = self._formChain(prompt)
        result = chain.invoke({"context": context, "answer": previousAnswer, "question": question})
        return result

    def setLang(self, lang):
        self.lang = lang