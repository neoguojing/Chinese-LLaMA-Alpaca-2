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
from apps.multi_task.seamless_m4t import SeamlessM4t
from apps.text2image.sd import StableDiff
from apps.config import model_root
from pydantic import  Field, root_validator
import torch


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
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:"]
    react_stop_words_tokens: Optional[List[List[int]]]
    

    def __init__(self, model_path: str,**kwargs):
        super(QwenLLM, self).__init__()
        self.model_path: str = model_path
        self.model,self.tokenizer = load_model(model_path=model_path,llama=False)
        self.react_stop_words_tokens = [self.tokenizer.encode(stop_) for stop_ in self.stop]

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

    @staticmethod
    def get_model(model_name,model_path=""):
        if model_name == "openai":
            return OpenAI()
        elif model_name == "claude":
            return ChatAnthropic()
        elif model_name == "qwen": 
            model_path = os.path.join(model_root,"chinese/Qwen-7B-Chat")
            return QwenLLM(model_path=model_path)
        elif model_name == "qianfan": 
            return QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
        elif model_name == "tongyi": 
            return Tongyi()
        elif model_name == "llama": 
            model_path = os.path.join(model_root,"chinese/chinese-alpaca-2-7b-hf")
            return LLamaLLM(model_path=model_path)
        elif model_name == "translate": 
            model_path = os.path.join(model_root,"nllb/")
            return Translate(model_path=model_path)
        elif model_name == "speech": 
            return SeamlessM4t()
        elif model_name == "text2image": 
            return StableDiff()
        else:
            raise Exception("Invalid model name")