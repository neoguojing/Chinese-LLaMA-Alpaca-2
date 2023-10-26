
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
    source: Optional[str]
    data: List[QAItem] = Field(..., description="问题答案列表")

    def merge_data(self, other: 'QAPackage'):
        self.data.extend(other.data)

    def length(self):
        return len(self.data)
    
    def dump(self):
        dict_obj = self.dict()
        return json.dumps(dict_obj.data, ensure_ascii=False, indent=4)
    
    def toQwen(self):
        qwen_package = QwenPackage()
        for qa in self.data:
            qwen_item = QwenItem(id=self.source,conversations=[])
            q = QwenConversationItem(from_="user",value=qa.question)
            a = QwenConversationItem(from_="assistant",value=qa.answer)
            qwen_item.conversations.append([q,a])
            qwen_package.data.append(qwen_item)
        return qwen_package.dump()
        


class QwenConversationItem(BaseModel):
    from_: str = Field(..., alias="from", description="发送方")
    value: str = Field(..., description="消息内容")


class QwenItem(BaseModel):
    id: str = Field(..., description="标识")
    conversations: List[QwenConversationItem] = Field(..., description="对话列表")

    def merge_data(self, other: 'QwenItem'):
        self.conversations.extend(other.conversations)

    def length(self):
        return len(self.conversations)

class QwenPackage(BaseModel):
    data: List[QwenItem] = Field(..., description="问题答案列表")

    def merge_data(self, other: 'QwenPackage'):
        self.data.extend(other.data)

    def length(self):
        return len(self.data)
    
    def dump(self):
        dict_obj = self.dict()
        return json.dumps(dict_obj.data, ensure_ascii=False, indent=4)

    
class JsonOutputParser(AgentOutputParser):
    pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)
    qaList: QAPackage = QAPackage(data=[])

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if the output contains valid JSON
        try:
            action_match = self.pattern.search(llm_output)
            if action_match is not None:
                response = action_match.group(1).strip()
                response = json.loads(response, strict=False)
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
            print("step qa num:",tmp.length())
            self.qaList.merge_data(tmp)
            print("total:",self.qaList.length())
        return AgentFinish(return_values=output,log=llm_output)
    
    def dump(self, path: str):
        print("final:",self.qaList.length())
        self.qaList.source = path
        with open(path+".json", 'w', encoding='utf-8') as f:
            package_json = self.qaList.dump()
            f.write(package_json)
        with open(path+".qwen", 'w', encoding='utf-8') as f:
            package_json = self.qaList.toQwen()
            f.write(package_json)
            
    def load(self, path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    