import sys
import os
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain.llms import OpenAI,Tongyi
from langchain.llms import HuggingFacePipeline
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from apps.inference import load_model,chat
from apps.translate.nllb import Translate
from apps.multi_task.speech import SeamlessM4t,Whisper,XTTS
from apps.image.sd import StableDiff,Image2Image
from apps.config import model_root
from apps.base import CustomerLLM
from pydantic import  Field, root_validator
import torch
import threading
import gc
import weakref


from langchain.chat_models import ChatAnthropic,QianfanChatEndpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# openai
os.environ['OPENAI_API_KEY'] = ''
# qianfan
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"
# tongyi
os.environ["DASHSCOPE_API_KEY"] = ""

class LLamaLLM(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    tokenizer: Any = None
    chat_format: Optional[str]   = 'llama'
    max_window_size: Optional[int]   = 3096

    def __init__(self, model_path: str,**kwargs):
        model,tokenizer = load_model(model_path=model_path,llama=True)
        super(LLamaLLM, self).__init__(llm=model)
        self.model_path: str = model_path
        self.tokenizer = tokenizer
        

        

    @property
    def _llm_type(self) -> str:
        return "llama"
    
    @property
    def model_name(self) -> str:
        return "llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response, _ = chat(self.model,self.tokenizer,prompt,history=None,
                           chat_format=self.chat_format,
                           max_window_size=self.max_window_size)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

class QwenLLM(CustomerLLM):
    model_path: str = Field(None, alias='model_path')

    tokenizer: Any = None
    chat_format: Optional[str]   = 'chatml'
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:"]
    react_stop_words_tokens: Optional[List[List[int]]]
    

    def __init__(self, model_path: str,**kwargs):
        model,tokenizer = load_model(model_path=model_path,llama=False,load_in_8bit=False)
        super(QwenLLM, self).__init__(llm=model)
        self.model_path: str = model_path
        self.tokenizer = tokenizer
        self.react_stop_words_tokens = [self.tokenizer.encode(stop_) for stop_ in self.stop]
        

    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    @property
    def model_name(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is None:
            self.react_stop_words_tokens.extend([self.tokenizer.encode(stop_) for stop_ in stop])
            
        response, _ = chat(self.model,self.tokenizer,prompt,history=None,
                           chat_format=self.chat_format,
                           max_window_size=self.max_window_size,
                           stop_words_ids=self.react_stop_words_tokens
                           )
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

class ModelFactory:
    temperature = 0
    _instances = {}
    _lock = threading.Lock()  # 异步锁

    @staticmethod
    def get_model(model_name,model_path=""):
        if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
            with ModelFactory._lock:
                if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
                    print(f"loading the model {model_name},wait a minute...")
                    if model_name == "openai":
                        instance = OpenAI()
                    elif model_name == "claude":
                        instance = ChatAnthropic()
                    elif model_name == "qwen": 
                        # model_path = os.path.join(model_root,"chinese/Qwen-7B-Chat")
                        model_path = os.path.join(model_root,"chinese/Qwen/Qwen-7B-Chat-Int4")
                        instance = QwenLLM(model_path=model_path)
                    elif model_name == "qianfan": 
                        instance = QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
                    elif model_name == "tongyi": 
                        instance = Tongyi()
                    elif model_name == "llama": 
                        model_path = os.path.join(model_root,"chinese/chinese-alpaca-2-7b-hf")
                        instance = LLamaLLM(model_path=model_path)
                    elif model_name == "translate": 
                        model_path = os.path.join(model_root,"nllb/")
                        instance = Translate(model_path=model_path)
                    elif model_name == "speech": 
                        instance = SeamlessM4t()
                    elif model_name == "text2image": 
                        instance = StableDiff()
                    elif model_name == "image2image": 
                        instance = Image2Image()
                    elif model_name == "speech2text": 
                        instance = Whisper()
                    elif model_name == "text2speech": 
                        instance = XTTS()
                    else:
                        raise Exception("Invalid model name")
                    
                    ModelFactory._instances[model_name] = instance
                    print(f"model {model_name} load finished...")
                    
        return  ModelFactory._instances[model_name]
    
    @staticmethod
    def destroy(model_name):
        if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
            with ModelFactory._lock:
                if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
                    obj = ModelFactory._instances.get(model_name)
                    refcount = len(gc.get_referrers(obj))
                    # print(f"{type(obj)} refer by {refcount} object")
                    if refcount <= 2:
                        ModelFactory._instances[model_name] = None
                        refcount = len(gc.get_referrers(obj))
                        # print(f"----{type(obj)} refer by {refcount} object")
                        if isinstance(obj, CustomerLLM) and refcount <= 1 :
                            obj.destroy()

    @staticmethod
    def release():
        for k, model in ModelFactory._instances.items():
            ModelFactory.destroy(k)