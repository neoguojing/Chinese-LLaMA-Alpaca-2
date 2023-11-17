import abc
import asyncio
import torch
from langchain.llms.base import LLM
from pydantic import  Field
from typing import Any
import time

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
            print("model destroy success")

    @property
    def model_name(self) -> str:
        return ""

class Task(ITask):
    _excurtor: CustomerLLM = None
    qinput = asyncio.Queue()
    qoutput: asyncio.Queue = None
    stop_event = asyncio.Event()

    def __init__(self,output:asyncio.Queue=None):
        self.qoutput = output

    def function_stats(self, func):
        call_stats = {"call_count": 0, "last_call_time": None}

        def wrapper(*args, **kwargs):
            nonlocal call_stats

            # 更新调用次数和时间
            call_stats["call_count"] += 1
            current_time = time.time()
            if call_stats["last_call_time"] is not None:
                elapsed_time = current_time - call_stats["last_call_time"]
                print(f"函数 {func.__name__} 上次调用时间间隔: {elapsed_time}秒")
            call_stats["last_call_time"] = current_time

            # 执行目标函数
            return func(*args, **kwargs)

        def get_call_count():
            return call_stats["call_count"]

        def get_last_call_time():
            return call_stats["last_call_time"]

        # 添加访问方法到装饰器函数对象
        wrapper.get_call_count = get_call_count
        wrapper.get_last_call_time = get_last_call_time

        # 返回装饰后的函数
        return wrapper
    
    def statistic(self):
        return self.run.get_call_count(),self.get_last_call_time()

    @function_stats
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
        self._excurtor = None


    def bind_model_name(self):
        if self._excurtor is not None:
            return self._excurtor.model_name
        return None


