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
from langchain.tools import BaseTool
import datetime
from scipy.io import wavfile
import sounddevice as sd
import pdb


class SeamlessM4t(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    processor: Any = None
    # src_lang: str = "eng_Latn" 
    # dst_lang: str = "zho_Hans"
    file_path: str = "./"
    sample_rate: Any = 16000
    save_to_file: bool = False

    def __init__(self, model_path: str = os.path.join(model_root,"seamless-m4t"),**kwargs):
        super(SeamlessM4t, self).__init__(llm=SeamlessM4TModel.from_pretrained(model_path))
        self.model_path = model_path
        # pdb.set_trace()
        # self.processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large",cache_dir=os.path.join(model_root,"seamless-m4t"))
        # self.model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large",cache_dir=os.path.join(model_root,"seamless-m4t"))
        # self.processor.save_pretrained(os.path.join(model_root,"seamless-m4t"))
        # self.model.save_pretrained(os.path.join(model_root,"seamless-m4t"))

        # processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
        # model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
        # processor.save_pretrained(os.path.join(model_root,"seamless-m4t-medium"))
        # model.save_pretrained(os.path.join(model_root,"seamless-m4t-medium"))

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("SeamlessM4t:device =",self.device)
        self.sample_rate = self.model.config.sampling_rate
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
        **kwargs: Any,
    ) -> str:
        generate_speech = kwargs.pop("generate_speech",True)
        src_lang = kwargs.pop("src_lang","eng")
        tgt_lang = kwargs.pop("tgt_lang","cmn")

        inputs = None
        if isinstance(prompt, str):
            inputs = self.processor(text=prompt, return_tensors="pt",src_lang=src_lang)
        else:
            # pdb.set_trace()
            inputs = self.processor(audios=[prompt.T],sampling_rate=self.sample_rate, return_tensors="pt")

        inputs.to(self.device)
        ret = ""
        if generate_speech:
            output = self.model.generate(**inputs, tgt_lang=tgt_lang,generate_speech=generate_speech)[0].cpu().numpy().squeeze()
            sd.play(output,self.sample_rate, blocking=False)
            if self.save_to_file:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = f"{now}_{self.tgt_lang}_{self.sample_rate}.wav"
                path = os.path.join(self.file_path, file_name)
                wavfile.write(path,rate=self.sample_rate, data=output)
                ret = path
        else:
            output = self.model.generate(**inputs, tgt_lang=tgt_lang,generate_speech=generate_speech)
            ret = self.processor.decode(output[0].tolist()[0], skip_special_tokens=True)
        return ret

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    

if __name__ == '__main__':
    sd = SeamlessM4t()
    sd.predict("Hello, my dog is cute")