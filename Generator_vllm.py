from langchain_community.llms import VLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

class Generator:
    def __init__(
            self,
            model="/data/public/wangshuo/LongContext/model/THUDM/LongAlign-13B-64k",
            num_gpus=2
        ):

        llm = VLLM(
            model=model,
            trust_remote_code=True,  # mandatory for hf models
            tensor_parallel_size=num_gpus,
            top_k=10,
            top_p=0.95,
            temperature=0.8
        )

        self.__llm = llm
        print("=====Create a summariser successfully!=====")


    def sum(self,text):
        promptTemplate = """Write a concise summary of the following text:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(promptTemplate)
        llm_chain = prompt | self.__llm
        sum=llm_chain.invoke({"text": text})
        print(sum)
        print("=====Summarise successfully!=====")
        return sum
    
    def ask(self, summary):
        promptTemplate = """Based on the following summary, please ask one question:
        "{summary}"
        QUESTION:"""
        prompt = PromptTemplate.from_template(promptTemplate)
        llm_chain = prompt | self.__llm
        question = llm_chain.invoke({"summary": summary})
        print(question)
        print("=====Ask successfully!=====")
        return question
    
    def answer(self, context, question):
        promptTemplate = """
        "{context}"
        Based on the above context, please answer the following question:
        "{question}"
        ANSWER:"""
        prompt = PromptTemplate.from_template(promptTemplate)
        llm_chain = prompt | self.__llm
        answer = llm_chain.invoke({"question": question, "context": context})
        print(answer)
        print("=====Answer successfully!=====")
        return answer

    