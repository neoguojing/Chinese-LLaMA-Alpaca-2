from typing import Any, List, Mapping, Optional,Dict
from pydantic import  Field, BaseModel,validator
import json
from pathlib import Path
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser

class QAItem(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

    
class QAPackage(BaseModel):
    data: List[QAItem] = Field(..., description="data")

class QwenConversationItem(BaseModel):
    from_: str = Field(..., alias="from", description="发送方")
    value: str = Field(..., description="消息内容")


class QwenItem(BaseModel):
    id: str = Field(..., description="标识")
    conversations: List[QwenConversationItem] = Field(..., description="对话列表")


class QwenPackage(BaseModel):
    data: List[QwenItem] = Field(..., description="问题答案列表")



class JsonOutputParser(AgentOutputParser):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def parse(self, llm_output: str) -> Dict[str, Any]:
        
        # Check if the output contains valid JSON
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON in LLM output {e}\n{llm_output}")
        
        # Parse the JSON into a dictionary
        output = {}
        if isinstance(data, dict):
            output = data
        elif isinstance(data, list):
            output = {"data": data}

        with open(self.file_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        return AgentFinish(return_values=output)