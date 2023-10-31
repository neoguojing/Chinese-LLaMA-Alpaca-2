
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

    def length(self):
        return len(self.data)
    
    def dump(self):
        dict_obj = self.dict()
        return json.dumps(dict_obj["data"], ensure_ascii=False, indent=4)
    
    def load(self, path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                json_data = json.load(f)
                qa_items = [QAItem(**item) for item in json_data]
                self.data=qa_items
        except FileNotFoundError:
            return None
        
    def toQwen(self,source:str):
        qwen_item = QwenItem(id=source,conversations=[])
        for qa in self.data:
            if qa.answer == "":
                continue
            data_input = {"from": "user", "value": qa.question}
            q = QwenConversationItem(**data_input)
            qwen_item.conversations.append(q)
            data_input = {"from": "assistant", "value": qa.answer}
            a = QwenConversationItem(**data_input)
            qwen_item.conversations.append(a)

        return qwen_item.dump()
    
    def toLLama(self,path,source:str):
        with open(path+source+"_llama"+".json", 'w', encoding='utf-8') as f:
            for qa in self.data:
                if qa.answer == "":
                    continue
                data_input = {"instruction": "", "input": qa.question,"output":qa.answer}
                q = LlamaItem(**data_input)
                json_str = q.dump()
                f.write(json_str+"\n")


        

class LlamaItem(BaseModel):
    instruction: str = Field(...,  description="指令")
    input: str = Field(..., description="输入")
    output: str = Field(..., description="输出")

    def dump(self):
        dict_obj = self.dict()
        return json.dumps(dict_obj, ensure_ascii=False, indent=2)


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
    
    def dump(self):
        dict_obj = self.dict()
        return json.dumps(dict_obj, ensure_ascii=False, indent=2)
    
    def toLLama(self,path,source:str):
        with open(path+source+"_llama"+".json", 'w', encoding='utf-8') as f:
            for i in range(0, len(self.conversations), 2):
                data_input = {"instruction": "", "input": self.conversations[i].value,"output":self.conversations[i+1].value}
                q = LlamaItem(**data_input)
                json_str = q.dump()
                f.write(json_str+"\n")

class JsonOutputParser(AgentOutputParser):
    pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)
    qaList: QAPackage = QAPackage(data=[])

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print("llm_output--------",llm_output)
        data = None
        # Check if the output contains valid JSON
        try:
            action_match = self.pattern.search(llm_output)
            if action_match is not None:
                response = action_match.group(1).strip()
            else:
                response = llm_output
            response = json.loads(response, strict=False)
            data = response
        except json.JSONDecodeError:
            print("***********Invalid JSON in LLM output")
        
        # Parse the JSON into a dictionary
        print("data-----",data)
        output = {}
        if isinstance(data, dict):
            output = data
            tmp = QAPackage(data=data["data"])
            print("step qa num:",tmp.length())
            self.qaList.merge_data(tmp)
            print("total:",self.qaList.length())
        elif isinstance(data, list):
            output = {"data": data}
            tmp = QAPackage(data=data)
            print("step qa num:",tmp.length())
            self.qaList.merge_data(tmp)
            print("total:",self.qaList.length())
        return AgentFinish(return_values=output,log=llm_output)
    
    def dump(self, path: str):
        print("final:",self.qaList.length())
        with open(path+".json", 'w', encoding='utf-8') as f:
            package_json = self.qaList.dump()
            f.write(package_json)
        with open(path+".qwen", 'w', encoding='utf-8') as f:
            package_json = self.qaList.toQwen(path)
            f.write(package_json)

        self.qaList = QAPackage(data=[])
            
    def load(self, path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        
if __name__ == '__main__':
    # qas = QAPackage(data=[])
    # qas.load("./ir2023_ashare.json")
    # qas.toLLama(".","ir2023_ashare")
    # output = qas.toQwen("ir2023_ashare.qw")
    # with open("ir2023_ashare.qw", 'w', encoding='utf-8') as f:
    #     f.write(output)
    with open("../dataset/chat/ir2023_ashare.qwen") as f:
        data = f.read()
        qw = QwenItem.parse_raw(data) 
        qw.toLLama("./","ir2023_ashare")
    