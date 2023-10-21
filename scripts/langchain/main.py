
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,TextLoader
from model_factory import  ModelFactory
import textwrap
verbose = True
if __name__ == '__main__':
    #map
    map_template = """以下是原始文本内容：
    {text}
    以问答的形式输出解析结果"""
    map_prompt = PromptTemplate(template=map_template,input_variables=["text"])

    # Reduce
    reduce_template = """依据以下问答对:
    {text}
    将问答内容去重"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # loader = DirectoryLoader('../../dataset/generate/', glob="**/*.txt",loader_cls=TextLoader)
    loader = TextLoader("./doc.txt")
    docs = loader.load()
    # print("**************",docs[0])
    
    # llm = ModelFactory().get_model("openai")
    llm = ModelFactory().get_model("llama")
    

    summary_chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, 
                                         combine_prompt=reduce_prompt,verbose=verbose)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain,
                                                    text_splitter=text_splitter)
    # docs[0].page_content
    qa = summarize_document_chain.run(docs[0].page_content)
    print(f"qa: {textwrap.fill(qa, width=100)}")