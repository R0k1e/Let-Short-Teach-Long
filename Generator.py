from langchain_community.llms import VLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import Generator_utils
from langchain_core.output_parsers import StrOutputParser
from GenerateFailedException import GenerateFailedException
from transformers import AutoTokenizer
import Generator_utils
import os, re
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

class Generator:
    def __init__(
            self,
            llm,
            tokenizer = None
        ):

        self.llm = llm
        self.tokenizer = tokenizer
        print("=====Create a summariser successfully!=====")


    def summary(self, textType, text):
        promptTemplate = Generator_utils.getPrompt("summary")
        prompt = PromptTemplate.from_template(promptTemplate)
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            llm_chain = prompt | self.llm | outputParser
        else:
            llm_chain = prompt | self.llm 
        
        try: 
            summary_result = llm_chain.invoke({"type": textType,"context": text})
        except Exception as e:
            print(e)
            raise GenerateFailedException("summary")
        print(summary_result)
        print("=====Summarise successfully!=====")
        return summary_result

    def ask(self, context) -> dict[str]:
        promptTemplate = Generator_utils.getPrompt("ask")
        questionCategories = Generator_utils.getPrompt("questionCategory")
        questionCategory = random.choice(questionCategories)
        comprehensions = Generator_utils.getPrompt("comprehension")
        comprehension = random.choice(comprehensions)
        prompt = PromptTemplate.from_template(promptTemplate)
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            llm_chain = prompt | self.llm | outputParser    
        else:
            llm_chain = prompt | self.llm 

        try:
            question = llm_chain.invoke({"questionCategory": questionCategory, "comprehension": comprehension, "context": context})
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        print(question)
        print("=====Ask successfully!=====")
        return {"question":question, "questionCategory":questionCategory, "comprehension":comprehension}

    
    def identifyType(self, context):
        promptTemplate = Generator_utils.getPrompt("identifyType")
        prompt = PromptTemplate.from_template(promptTemplate)
        
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            llm_chain = prompt | self.llm | outputParser
        else:
            llm_chain = prompt | self.llm 

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


    def mr_map(self, context, question):
        map_template = Generator_utils.getPrompt("answer")
        map_prompt = PromptTemplate.from_template(map_template)
        
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            map_chain = map_prompt | self.llm | outputParser
        else:
            map_chain = map_prompt | self.llm  

        answer = ""
        answerList = {}
        print("=====Map=====")
        for i, item in enumerate(context):
            result = map_chain.invoke({"question": question, "context": item})
            print(f"Chunk {i+1}/{len(context)}:\n{result}")
            answer += result
            answerList[f"Chunk {i+1}"] = result
        
        return answer, answerList
        
    def mr_reduce(self, answers, question):
        # Reduce
        reduce_template = Generator_utils.getPrompt("reduce")
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        
        if isinstance(self.llm, ChatOpenAI):
            outputParser = StrOutputParser()
            reduce_chain = reduce_prompt | self.llm | outputParser 
        else:
            reduce_chain = reduce_prompt | self.llm 
        print("=====Reduce=====")
        result = reduce_chain.invoke({"answer": answers, "question": question})
        print(result)
        return result
