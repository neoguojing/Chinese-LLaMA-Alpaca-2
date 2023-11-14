import abc
import asyncio
import torch
from langchain.llms.base import LLM
from pydantic import  Field

class ITask(abc.ABC):
    
    @abc.abstractmethod
    def run(self):
        pass

class Task(ITask):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')

    def run(self):
        print("task")



class CustomerLLM(LLM):
    device: str = Field(torch.device('cpu'))
    def __init__(self,**kwargs):
        super(CustomerLLM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')