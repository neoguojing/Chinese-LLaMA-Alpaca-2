
import os

os.environ['OPENAI_API_KEY'] = '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 '
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,TextLoader

if __name__ == '__main__':
    prompt_template = """解析以下文件内容：


    {text}


    以问答的形式输出解"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    loader = DirectoryLoader('../../dataset/generate/', glob="**/*.txt",loader_cls=TextLoader)
    docs = loader.load()
    # print("**************",docs[0])
    
    llm = OpenAI(temperature=0)

    summary_chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain,text_splitter=text_splitter)
    summarize_document_chain.run(docs[0].page_content)