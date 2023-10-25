
import json
from pathlib import Path
import re
from typing import Union
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser




class JsonOutputParser(AgentOutputParser):
    pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if the output contains valid JSON
        try:
            action_match = self.pattern.search(llm_output)
            if action_match is not None:
                response = action_match.group(1).strip()
                response = json.loads(response, strict=False)
                print("llm_output:",response)
                data = response
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in LLM output")
        
        # Parse the JSON into a dictionary
        output = {}
        if isinstance(data, dict):
            output = data
        elif isinstance(data, list):
            output = {"data": data}
            
        return AgentFinish(return_values=output,log=llm_output)