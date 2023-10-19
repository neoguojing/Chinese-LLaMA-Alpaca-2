
import os

os.environ['OPENAI_API_KEY'] = ''
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

if __name__ == '__main__':
    prompt_template = """Write a concise summary of the following:


    {text}


    CONCISE SUMMARY IN INDONESIAN:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with open("/content/article.txt") as f:
        article_kompas = f.read()

    llm = OpenAI(temperature=0)
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain,text_splitter=text_splitter)
    summarize_document_chain.run(article_kompas)