from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain, MapRerankDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.output_parsers.regex import RegexParser
import Generator_utils
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from GenerateFailedException import GenerateFailedException
import os, re
import random

os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''


class Generator:
    def __init__(
            self,
            model="gpt-3.5-turbo-1106",
        ):

        llm = ChatOpenAI(temperature=0, model_name=model)


        self.__llm = llm
        print("=====Create a summariser successfully!=====")


    def sum(self, type, text):
        promptTemplate = Generator_utils.getPrompt("sum")
        prompt = PromptTemplate.from_template(promptTemplate)
        outputParser = StrOutputParser()
        llm_chain = prompt | self.__llm | outputParser
        try: 
            sum = llm_chain.invoke({"type": type,"context": text})
        except Exception as e:
            print(e)
            raise GenerateFailedException("sum")
        print(sum)
        print("=====Summarise successfully!=====")
        return sum

    def ask(self, context) -> dict[str]:
        promptTemplate = Generator_utils.getPrompt("ask")
        questionCategories = Generator_utils.getPrompt("questionCategory")
        questionCategory = random.choice(questionCategories)
        comprehensions = Generator_utils.getPrompt("comprehension")
        comprehension = random.choice(comprehensions)
        prompt = PromptTemplate.from_template(promptTemplate)
        outputParser = StrOutputParser()
        llm_chain = prompt | self.__llm | outputParser
        try:
            question = llm_chain.invoke({"questionCategory": questionCategory, "comprehension": comprehension, "context": context})
        except Exception as e:
            print(e)
            raise GenerateFailedException("ask")
        print(question)
        print("=====Ask successfully!=====")
        return {"question":question, "questionCategory":questionCategory, "comprehension":comprehension}
    
    # def answer(self, context, question):
    #     promptTemplate = """
    #     "{context}"
    #     Based on the above context, please answer the following question:
    #     "{question}"
    #     ANSWER:"""
    #     prompt = PromptTemplate.from_template(promptTemplate)
    #     outputParser = StrOutputParser()
    #     llm_chain = prompt | self.__llm | outputParser
    #     try:
    #         answer = llm_chain.invoke({"question": question, "context": context})
    #     except Exception as e:
    #         print(e)
    #         raise GenerateFailedException("answer")
    #     print(answer)
    #     print("=====Answer successfully!=====")
    #     return answer

    
    # def mapReduce(self, context, question):
    #     question=f"{question}\nHelpful Answer:"
    #     map_template = """The following is a summary of a part of a document:
    #     {context}
    #     Based on this summary, please answer the following question: 
    #     """+ question
    #     map_prompt = PromptTemplate.from_template(map_template)
    #     map_chain = LLMChain(llm=self.__llm, prompt=map_prompt)
    #     # Reduce
    #     reduce_template = """The following is set of answers:
    #     {context}
    #     Take these and distill it into a final, consolidated answer for this question:
    #     """+ question
    #     reduce_prompt = PromptTemplate.from_template(reduce_template)
    #     # Run chain
    #     reduce_chain = LLMChain(llm=self.__llm, prompt=reduce_prompt)

    #     # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    #     # llm_chain is a destination
    #     combine_documents_chain = StuffDocumentsChain(
    #         llm_chain=reduce_chain, document_variable_name="context"
    #     )

    #     # Combines and iteratively reduces the mapped documents
    #     # combine_documents_chain is a destination
    #     # collapse_documents_chain copes with the exceeding context
    #     reduce_documents_chain = ReduceDocumentsChain(
    #         # This is final chain that is called.
    #         combine_documents_chain=combine_documents_chain,
    #         # If documents exceed context for `StuffDocumentsChain`
    #         collapse_documents_chain=combine_documents_chain,
    #         # The maximum number of tokens to group documents into.
    #         token_max=4000,
    #     )

    #     # Combining documents by mapping a chain over them, then combining results
    #     map_reduce_chain = MapReduceDocumentsChain(
    #         # Map chain
    #         llm_chain=map_chain,
    #         # Reduce chain
    #         reduce_documents_chain=reduce_documents_chain,
    #         # The variable name in the llm_chain to put the documents in
    #         document_variable_name="context",
    #         # Return the results of the map steps in the output
    #         return_intermediate_steps=False,
    #         verbose=True
    #     )
    #     context = Generator_utils.docParser(context)
    #     outputParser = StrOutputParser()
    #     chain = map_reduce_chain | outputParser
    #     try:
    #         result = chain.invoke(context)
    #     except Exception as e:
    #         print(e)
    #         raise GenerateFailedException("mapReduce")
    #     return result
    
    # def refine(self, context, question):
    #     question_prompt=f"{question}\nHelpful Answer:"
    #     # This controls how each document will be formatted. Specifically,
    #     # it will be passed to `format_document` - see that function for more
    #     # details.
    #     document_prompt = PromptTemplate(
    #         input_variables=["page_content"],
    #         template="{page_content}"
    #     )
    #     document_variable_name = "context"
    #     # The prompt here should take as an input variable the
    #     # `document_variable_name`
    #     prompt = PromptTemplate.from_template(
    #         """The following is a summary of a part of a document:
    #     {context}
    #     Based on this summary, please answer the following question: """+question_prompt
    #     )
    #     initial_llm_chain = LLMChain(llm=self.__llm, prompt=prompt)
    #     initial_response_name = "prev_response"
    #     # The prompt here should take as an input variable the
    #     # `document_variable_name` as well as `initial_response_name`
    #     prompt_refine = PromptTemplate.from_template(
    #         "Here's your first answer: {prev_response}. "+"Here is the question: "+question
    #         +"Now refine it based on the following context: {context}"
    #     )
    #     refine_llm_chain = LLMChain(llm=self.__llm, prompt=prompt_refine)
    #     chain = RefineDocumentsChain(
    #         initial_llm_chain=initial_llm_chain,
    #         refine_llm_chain=refine_llm_chain,
    #         document_prompt=document_prompt,
    #         document_variable_name=document_variable_name,
    #         initial_response_name=initial_response_name,
    #         return_intermediate_steps=True,
    #         verbose=True,
    #     )
    #     context = Generator_utils.docParser(context)
    #     try:
    #         result = chain.invoke(context)
    #     except Exception as e:
    #         print(e)
    #         raise GenerateFailedException("refine")
    #     return str(result)
        

    # def mapRerank(self, context, question):
    #     document_variable_name = "context"
    #     # The prompt here should take as an input variable the
    #     # `document_variable_name`
    #     # The actual prompt will need to be a lot more complex, this is just
    #     # an example.
    #     prompt_template = (
    #         "Use the following context to answer the following question: "+question
    #         +"Output both your answer and a score of how confident "
    #         +"you are. Context: {context} Remember that you must answer like this 'Answer: ... Score: ..."
    #     )
    #     output_parser = RegexParser(
    #         regex=r"(.*?)Score: (.*)",
    #         output_keys=["answer", "score"]
    #     ) 
    #     prompt = PromptTemplate(
    #         template=prompt_template, 
    #         input_variables=["context"], 
    #         output_parser=output_parser
    #     ) 
    #     llm_chain = LLMChain(llm=self.__llm, prompt=prompt) 
    #     chain = MapRerankDocumentsChain(
    #         llm_chain=llm_chain, 
    #         document_variable_name=document_variable_name, 
    #         rank_key="score", 
    #         answer_key="answer",
    #         return_intermediate_steps=True,
    #         verbose=True,
    #     )
    #     context = Generator_utils.docParser(context)
    #     try:
    #         result = chain.invoke(context)
    #     except Exception as e:
    #         print(e)
    #         raise GenerateFailedException("refine")
    #     print(type(result))
    #     return str(result)
    

    def identifyType(self, context):
        promptTemplate = Generator_utils.getPrompt("identifyType")
        prompt = PromptTemplate.from_template(promptTemplate)
        outputParser = StrOutputParser()
        llm_chain = prompt | self.__llm | outputParser
        try:
            result = llm_chain.invoke({"context": context})
        except Exception as e:
            print(e)
            raise GenerateFailedException("identifyType")
        print(result)
        match = re.search(r'Type:\s*(.+)', result)
        if match:
            type_name = match.group(1).strip(" .\n")
            print(type_name)
            return type_name
        else:
            print("Type Name not found.")
            raise GenerateFailedException("identifyType")

    
    # def metaQuestion(self, context, type):
    #     promptTemplate = Generator_utils.getPrompt("metaQuestion")
    #     prompt = PromptTemplate.from_template(promptTemplate)
    #     llm_chain = prompt | self.__llm
    #     result = llm_chain.invoke({"context": context})
    #     print(result)

    def mapReduceNew(self, context, question):
        map_template = Generator_utils.getPrompt("answer")
        map_prompt = PromptTemplate.from_template(map_template)
        outputParser = StrOutputParser()
        map_chain = map_prompt | self.__llm | outputParser
        answer = ""
        answerList = {}
        print("=====Map=====")
        for i, item in enumerate(context):
            result = map_chain.invoke({"question": question, "context": item})
            print(f"Chunk {i+1}/{len(context)}:\n{result}")
            answer += result
            answerList[f"Chunk {i+1}"] = result
        # Reduce
        reduce_template = """The following is set of answers:
        {answer}
        Take these and distill it into a final, consolidated answer for this question:
        {question}"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        
        reduce_chain = reduce_prompt | self.__llm | outputParser
        print("=====Reduce=====")
        result = reduce_chain.invoke({"answer": answer, "question": question})
        print(result)
        return result, answerList