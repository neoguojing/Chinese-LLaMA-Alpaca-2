
import os
import sys
import time
from parser import QAPackage,JsonOutputParser
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser
from langchain.chains import create_extraction_chain
from apps.model_factory import ModelFactory
from apps.persist import Persister

import textwrap
verbose = True
if __name__ == '__main__':

    
    # loader = DirectoryLoader('../../dataset/generate/', glob="**/*.txt",loader_cls=TextLoader)
    loader = TextLoader("./doc.txt")
    docs = loader.load()

    # llm = ModelFactory().get_model("openai")
    # llm = ModelFactory().get_model("claude")
    # llm = ModelFactory().get_model("qwen")
    llm = ModelFactory().get_model("qianfan")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    
    qaParser = PydanticOutputParser(pydantic_object=QAPackage)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "将\n{text}转换中文问答对，格式如下：\n{format_instructions}"
            )
        ],
        input_variables=["text"],
        partial_variables={
            "format_instructions": qaParser.get_format_instructions(),
        },
    )

    fixParser = OutputFixingParser.from_llm(parser=qaParser, llm=llm)
    jsonParser = JsonOutputParser()
    persist = Persister()
    chain = prompt | llm | jsonParser | persist
    
    texts = []
    for doc in docs:
        text = doc.page_content
        print(doc.metadata)
        texts += text_splitter.create_documents([text])
        for text in texts:
            print(text)
            answer = chain.invoke({"text": text,"format_instructions":qaParser.get_format_instructions()})
            print(f"Output: {answer}")
            time.sleep(1)
        persist.dump("doc.json")