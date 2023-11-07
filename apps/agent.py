from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chains.llm import LLMChain

from typing import Dict, Tuple
import os
import json

os.environ['SERPAPI_API_KEY'] = '重要！请在这里填入您的 SERPAPI_API_KEY！'
os.environ['WOLFRAM_ALPHA_APPID'] = '重要！请在这里填入您的 WOLFRAM_ALPHA_APPID！'

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
    },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "A Python shell. Use this to execute python commands. When using this tool, sometimes output is abbreviated - Make sure \
                it does not look abbreviated before using it in your answer. "
            "Don't add comments to your python code.",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "a valid python command.",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(python)
    }

]




def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''

def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    if use_toolname == "":
        return "no tool founds"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "no tool founds"
    
    api_output = used_tool_meta[0]["tool_api"](action_input)
    return api_output



if __name__ == '__main__':
    from .model_factory import ModelFactory
    from .prompt import QwenAgentPromptTemplate
    from .parser import QwenAgentOutputParser
    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    # prompt_1 = build_planning_prompt(TOOLS[0:1], query="加拿大2023年人口统计数字是多少？")
    # print(prompt_1)

    # api_output = use_api(TOOLS, response_1)
    # print(api_output)

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

    agent_executor.run("Search for Leo DiCaprio's girlfriend on the internet.")
    # agent = initialize_agent(
    #     [tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    # )
