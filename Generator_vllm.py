from langchain_community.llms import VLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain, MapRerankDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.output_parsers.regex import RegexParser
import Generator_utils
from langchain_core.output_parsers import StrOutputParser
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

    def mapReduce(self, context, question):
        question=f"{question}\nHelpful Answer:"
        map_template = """The following is a summary of a part of a document:
        {context}
        Based on this summary, please answer the following question: 
        """+ question
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.__llm, prompt=map_prompt)
        # Reduce
        reduce_template = """The following is set of answers:
        {context}
        Take these and distill it into a final, consolidated answer for this question:
        """+ question
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        # Run chain
        reduce_chain = LLMChain(llm=self.__llm, prompt=reduce_prompt)

        # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
        # llm_chain is a destination
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="context"
        )

        # Combines and iteratively reduces the mapped documents
        # combine_documents_chain is a destination
        # collapse_documents_chain copes with the exceeding context
        reduce_documents_chain = ReduceDocumentsChain(
            # This is final chain that is called.
            combine_documents_chain=combine_documents_chain,
            # If documents exceed context for `StuffDocumentsChain`
            collapse_documents_chain=combine_documents_chain,
            # The maximum number of tokens to group documents into.
            token_max=4000,
        )

        # Combining documents by mapping a chain over them, then combining results
        map_reduce_chain = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="context",
            # Return the results of the map steps in the output
            return_intermediate_steps=False,
            verbose=True
        )
        context = Generator_utils.docParser(context)
        return map_reduce_chain.run(context)
    
    def refine(self, context, question):
        question_prompt=f"{question}\nHelpful Answer:"
        # This controls how each document will be formatted. Specifically,
        # it will be passed to `format_document` - see that function for more
        # details.
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
        document_variable_name = "context"
        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt = PromptTemplate.from_template(
            """The following is a summary of part of a document:
        {context}
        Based on this summary, please answer the following question: """+question_prompt
        )
        initial_llm_chain = LLMChain(llm=self.__llm, prompt=prompt)
        initial_response_name = "prev_response"
        # The prompt here should take as an input variable the
        # `document_variable_name` as well as `initial_response_name`
        prompt_refine = PromptTemplate.from_template(
            "Here's your first answer: {prev_response}. "+"Here is the question: "+question
            +"Now refine it based on the following context: {context}"
        )
        refine_llm_chain = LLMChain(llm=self.__llm, prompt=prompt_refine)
        chain = RefineDocumentsChain(
            initial_llm_chain=initial_llm_chain,
            refine_llm_chain=refine_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
            initial_response_name=initial_response_name,
            return_intermediate_steps=True,
            verbose=True,
        )
        context = Generator_utils.docParser(context)
        result = chain.invoke(context)
        return str(result)
        

    def mapRerank(self, context, question):
        document_variable_name = "context"
        # The prompt here should take as an input variable the
        # `document_variable_name`
        # The actual prompt will need to be a lot more complex, this is just
        # an example.
        prompt_template = (
            "Use the following context to answer the following question: "+question
            +"Output both your answer and a score of how confident "
            +"you are. Context: {context} Remember that you need to answer like this 'Answer: ... Score: ...'"
        )
        output_parser = RegexParser(
            regex=r"(.*?)Score: (.*)",
            output_keys=["answer", "score"]
        ) 
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context"], 
            output_parser=output_parser
        ) 
        llm_chain = LLMChain(llm=self.__llm, prompt=prompt) 
        chain = MapRerankDocumentsChain(
            llm_chain=llm_chain, 
            document_variable_name=document_variable_name, 
            rank_key="score", 
            answer_key="answer",
            return_intermediate_steps=False,
            verbose=True,
        )
        context = Generator_utils.docParser(context)
        result= chain.invoke(context)
        print(type(result))
        return str(result)