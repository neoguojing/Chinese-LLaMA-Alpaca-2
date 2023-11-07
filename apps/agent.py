from langchain import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities import ArxivAPIWrapper
from langchain.agents import AgentType, initialize_agent

from typing import Dict, Tuple
import os
import json

os.environ['SERPAPI_API_KEY'] = '重要！请在这里填入您的 SERPAPI_API_KEY！'
os.environ['WOLFRAM_ALPHA_APPID'] = '重要！请在这里填入您的 WOLFRAM_ALPHA_APPID！'

search = SerpAPIWrapper()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
python=PythonAstREPLTool()

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

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful \
    for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""

def build_planning_prompt(TOOLS, query):
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt



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
    prompt_1 = build_planning_prompt(TOOLS[0:1], query="加拿大2023年人口统计数字是多少？")
    print(prompt_1)

    api_output = use_api(TOOLS, response_1)
    print(api_output)
    
    llm = ModelFactory().get_model("qwen")
    agent = initialize_agent(
        [tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    )
