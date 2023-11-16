import abc
import asyncio
import torch
from langchain.llms.base import LLM
from pydantic import  Field
from typing import Any

class ITask(abc.ABC):
    
    @abc.abstractmethod
    def run(self,input:str):
        pass
    
    @abc.abstractmethod
    def init_model(self):
        pass

class CustomerLLM(LLM):
    device: str = Field(torch.device('cpu'))
    model: Any = None
    def __init__(self,llm,**kwargs):
        super(CustomerLLM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        self.model = llm

    def destroy(self):
        if self.model is not None:
            del self.model

class Task(ITask):
    _excurtor: CustomerLLM = None
    qinput = asyncio.Queue()
    qoutput: asyncio.Queue = None
    stop_event = asyncio.Event()

    def __init__(self,output:asyncio.Queue=None):
        self.qoutput = output

    def run(self,input:str):
        output = self.excurtor.predict(input)
        return output

    @property
    def excurtor(self):
        if self._excurtor is None:
            # 执行延迟初始化逻辑
            self._excurtor = self.init_model()
        return self._excurtor
    
    def init_model(self):
        return None
    
    def input(self,input:str):
        self.qinput.put_nowait(input)

    # async def arun(self):
    #     self.stop_event.clear()
    #     while not self.stop_event.is_set():
    #         _input = self.qinput.get()
    #         result = self.excurtor.predict(_input)
    #         self.qoutput.put_nowait(result)

    def destroy(self):
        self.stop_event.set()


