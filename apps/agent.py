
from langchain.agents import AgentType, initialize_agent
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from model_factory import ModelFactory
from prompt import QwenAgentPromptTemplate
from parser import QwenAgentOutputParser
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import Dict, Tuple
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from apps.base import Task,CustomerLLM
from apps.tools import tools 

class Agent(Task):
    def __init__(self):
        prompt = QwenAgentPromptTemplate(
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )

        output_parser = QwenAgentOutputParser()
        llm_chain = LLMChain(llm=self.excurtor, prompt=prompt)

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    def run(self,input: str=None):
        output = self.agent_executor.run(input)
        return output
    
    def init_model(self):
        model = ModelFactory.get_model("qwen")
        return model
    

# if __name__ == '__main__':
    
