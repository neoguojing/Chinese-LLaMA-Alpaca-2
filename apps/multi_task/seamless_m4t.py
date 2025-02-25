import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from transformers import AutoProcessor, SeamlessM4TModel
import torch
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import  Field
from apps.base import Task,CustomerLLM
from apps.config import model_root
import random
import hashlib
from langchain.tools import BaseTool
import pdb

def calculate_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode('utf-8'))
    md5_value = md5_hash.hexdigest()
    return md5_value

class SeamlessM4t(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    processor: Any = None
    src_lang: str = "eng_Latn" 
    dst_lang: str = "zho_Hans"
    file_path: str = "./"

    def __init__(self, model_path: str = os.path.join(model_root,"seamless-m4t"),**kwargs):
        super(SeamlessM4t, self).__init__(llm=SeamlessM4TModel.from_pretrained(model_path))
        self.model_path = model_path
        # pdb.set_trace()
        # self.processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large",cache_dir=os.path.join(model_root,"seamless-m4t"))
        # self.model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large",cache_dir=os.path.join(model_root,"seamless-m4t"))
        # self.processor.save_pretrained(os.path.join(model_root,"seamless-m4t"))
        # self.model.save_pretrained(os.path.join(model_root,"seamless-m4t"))

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("SeamlessM4t:device =",self.device)
        self.model.to(self.device)
        
        

    @property
    def _llm_type(self) -> str:
        return "facebook/hf-seamless-m4t-large"
    
    @property
    def model_name(self) -> str:
        return "speech"
    
    def _call(
        self,
        prompt: Union[str,any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_speech:Optional[bool] = False,
        tgt_lang:Optional[str] = "cmn",
        src_lang:Optional[str] = "eng",
        **kwargs: Any,
    ) -> str:
        inputs = None
        if isinstance(prompt, str):
            inputs = self.processor(text=prompt, return_tensors="pt",src_lang=src_lang)
        else:
            inputs = self.processor(audios=prompt, return_tensors="pt")

        inputs.to(self.device)
        
        ret = ""
        if generate_speech:
            output = self.model.generate(**inputs, tgt_lang=tgt_lang,generate_speech=generate_speech)[0].cpu().numpy().squeeze()
            file_name = calculate_md5(prompt)
            path = os.path.join(self.file_path, file_name)
            with open(path, 'wb') as file:
                for audio in output:
                    file.write(audio)
            ret = path
        else:
            output = self.model.generate(**inputs, tgt_lang=tgt_lang,generate_speech=generate_speech)
            ret = self.processor.decode(output[0].tolist(), skip_special_tokens=True)
        return ret

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    

# if __name__ == '__main__':
#     sd = SeamlessM4t()
#     sd.predict("Hello, my dog is cute")