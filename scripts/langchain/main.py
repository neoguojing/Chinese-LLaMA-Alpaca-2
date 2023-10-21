
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.chains.question_answering import load_qa_chain
from model_factory import  ModelFactory
import textwrap
verbose = True
if __name__ == '__main__':
 
    prompt_str = """
        将如下文字转化为问答： 
        {text}
    """
    prompt = PromptTemplate.from_template(prompt_str)

    # loader = DirectoryLoader('../../dataset/generate/', glob="**/*.txt",loader_cls=TextLoader)
    loader = TextLoader("./doc.txt")
    docs = loader.load()
    # print("**************",docs[0])
    
    # llm = ModelFactory().get_model("openai")
    llm = ModelFactory().get_model("qwen")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    texts = []
    for doc in docs:
        # print(doc.page_content)
        text = doc.page_content
        texts += text_splitter.create_documents([text])

    chain = prompt | llm 
    
    for text in texts:
        print(text)
        answer = chain.invoke({"text": text})
        print(f"answer: {textwrap.fill(answer, width=100)}")