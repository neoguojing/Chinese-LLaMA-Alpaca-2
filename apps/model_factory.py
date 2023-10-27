from langchain.llms import OpenAI,Tongyi
from langchain.llms import HuggingFacePipeline
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from apps.inference import load_model,chat
from pydantic import  Field, root_validator
import torch
import os
from langchain.chat_models import ChatAnthropic,QianfanChatEndpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# openai
os.environ['OPENAI_API_KEY'] = ''
# qianfan
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"
# tongyi
os.environ["DASHSCOPE_API_KEY"] = ""
class LLamaLLM(LLM):
    model_path: str = Field(None, alias='model_path')

    model: Any = None 
    tokenizer: Any = None
    chat_format: Optional[str]   = 'llama'
    max_window_size: Optional[int]   = 3096

    def __init__(self, model_path: str,**kwargs):
        super(LLamaLLM, self).__init__()
        self.model_path: str = model_path
        self.model,self.tokenizer = load_model(model_path=model_path,llama=True)

    @property
    def _llm_type(self) -> str:
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

class QwenLLM(LLM):
    model_path: str = Field(None, alias='model_path')

    model: Any = None 
    tokenizer: Any = None
    chat_format: Optional[str]   = 'chatml'
    max_window_size: Optional[int]   = 6144

    def __init__(self, model_path: str,**kwargs):
        super(QwenLLM, self).__init__()
        self.model_path: str = model_path
        self.model,self.tokenizer = load_model(model_path=model_path,llama=False)

    @property
    def _llm_type(self) -> str:
        return "qwen"

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

class ModelFactory:
    temperature = 0

    @staticmethod
    def get_model(model_name,model_path=""):
        if model_name == "openai":
            return OpenAI()
        elif model_name == "claude":
            return ChatAnthropic()
        elif model_name == "qwen": 
            model_path = "../model/chinese/Qwen-7B-Chat"
            return QwenLLM(model_path=model_path)
        elif model_name == "qianfan": 
            return QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
        elif model_name == "tongyi": 
            return Tongyi()
        elif model_name == "llama": 
            model_path = "../model/chinese/chinese-alpaca-2-7b-hf"
            return LLamaLLM(model_path=model_path)
        else:
            raise Exception("Invalid model name")