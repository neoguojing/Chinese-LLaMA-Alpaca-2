import json
from typing import Optional
from typing import Any, List, Mapping, Optional,Dict,Union
from pydantic import  Field, BaseModel,validator,ListModel
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


class Persister(AgentOutputParser):
    qaList = QAPackage(data=[])
    def dump(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.qaList, f)
            
    def load(self, path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            print("llm_output:",llm_output)
            tmp = QAPackage(data=llm_output)
            self.qaList.merge_data(tmp)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in LLM output")
        
        return AgentFinish(return_values="success",log=llm_output)