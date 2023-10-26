
import json
from pathlib import Path
import re
from typing import Any, List, Mapping, Optional,Dict,Union
from pydantic import  Field, BaseModel,validator
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser



class QAItem(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

    
class QAPackage(BaseModel):
    data: List[QAItem] = Field(..., description="问题答案列表")

    def merge_data(self, other: 'QAPackage'):
        self.data.extend(other.data)

class QwenConversationItem(BaseModel):
    from_: str = Field(..., alias="from", description="发送方")
    value: str = Field(..., description="消息内容")


class QwenItem(BaseModel):
    id: str = Field(..., description="标识")
    conversations: List[QwenConversationItem] = Field(..., description="对话列表")

    def merge_data(self, other: 'QwenItem'):
        self.conversations.extend(other.conversations)


class QwenPackage(BaseModel):
    data: List[QwenItem] = Field(..., description="问题答案列表")

    def merge_data(self, other: 'QwenPackage'):
        self.data.extend(other.data)

    
class JsonOutputParser(AgentOutputParser):
    pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)
    qaList = QAPackage(data=[])

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
            tmp = QAPackage(data=data)
            self.qaList.merge_data(tmp)
            
        return AgentFinish(return_values=output,log=llm_output)
    
    def dump(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.qaList, f)
            
    def load(self, path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None