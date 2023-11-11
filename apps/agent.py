from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chains.llm import LLMChain
from model_factory import ModelFactory
from prompt import QwenAgentPromptTemplate
from parser import QwenAgentOutputParser
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import Dict, Tuple
import os
import json

os.environ['SERPAPI_API_KEY'] = 'f765e0536e1a72c2f353bb1946875937b3ac7bed0270881f966d4147ac0a7943'
os.environ['WOLFRAM_ALPHA_APPID'] = 'QTJAQT-UPJ2R3KP89'

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Math",
        func=WolframAlpha.run,
        description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life."
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    )
]

def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        return tool.run(query)
    return tool_

# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google search',
        'name_for_model':
            'Search',
        'description_for_model':
            'useful for when you need to answer questions about current events.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "search query of google",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(search)
    },
    {
        'name_for_human':
            'Wolfram Alpha',
        'name_for_model':
            'Math',
        'description_for_model':
            'Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the problem to solved by Wolfram Alpha",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(WolframAlpha)
    },  
    {
        'name_for_human':
            'arxiv',
        'name_for_model':
            'Arxiv',
        'description_for_model':
            'A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org.',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "the document id of arxiv to search",
            'required': True
        }], 
        'tool_api': tool_wrapper_for_qwen(arxiv)
    }

]


if __name__ == '__main__':
    prompt = QwenAgentPromptTemplate(
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    llm = ModelFactory().get_model("qwen")
    output_parser = QwenAgentOutputParser()
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # output = agent_executor.run("Search for Leo DiCaprio's girlfriend on the internet.")
    output = agent_executor.run("今天北京的PM2.5")
    print(output)
    # 
    # agent = initialize_agent(
    #     [tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    # )
