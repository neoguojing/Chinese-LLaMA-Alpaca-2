import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from transformers import AutoProcessor, SeamlessM4TModel,AutoModelForSpeechSeq2Seq,pipeline
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
import librosa
import numpy as np



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
        self.sample_rate = self.model.config.sampling_rate
        self.model.to(self.device)
        print(f"SeamlessM4t:device ={self.device},sample_rate={self.sample_rate}")
        
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
            print("***********",prompt.shape)
            print("***********",prompt.T.shape)
            inputs = self.processor(audios=[prompt.T],sampling_rate=self.sample_rate, return_tensors="pt")

        inputs.to(self.device)
        ret = ""
        if generate_speech:
            output = self.model.generate(**inputs, tgt_lang=tgt_lang,generate_speech=generate_speech,spkr_id=4,
                                          num_beams=5, speech_do_sample=True, speech_temperature=0.6)[0].cpu().numpy().squeeze()
            print("SeamlessM4t video shape:",output.shape)
            output *= 1.2 # 增大音量
            output = np.reshape(output, (-1, 1))
            # print("2d output",output.shape)
            # output = librosa.resample(output, orig_sr=self.sample_rate, target_sr=44100) #增加采样率
            # print("resample output",output.shape)
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
    

class Whisper(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    processor: Any = None
    file_path: str = "./"
    sample_rate: Any = 16000
    save_to_file: bool = False
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe: Any  = None

    def __init__(self, model_path: str = os.path.join(model_root,"whisper"),**kwargs):
        super(Whisper, self).__init__(llm=AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            low_cpu_mem_usage=True, use_safetensors=True,use_flash_attention_2=True
        ))
        self.model_path = model_path
        # model_id = "openai/whisper-large-v3"
        # model = AutoModelForSpeechSeq2Seq.from_pretrained(
        #     "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
        # )
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.processor.save_pretrained(os.path.join(model_root,"whisper"))
        # model.save_pretrained(os.path.join(model_root,"whisper"))
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Whisper:device ={self.device},sample_rate={self.sample_rate}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        
    @property
    def _llm_type(self) -> str:
        return "openai/whisper-large-v3"
    
    @property
    def model_name(self) -> str:
        return "speech2text"
    
    def _call(
        self,
        prompt: Union[str,any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        generate_speech = kwargs.pop("language","mandarin")
        if isinstance(prompt, str):
            return prompt
        print(prompt.shape)
        prompt= np.squeeze(prompt)
        print(prompt.shape)
        result = self.pipe(prompt)
        return result["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
config = XttsConfig()
config.load_json(os.path.join(model_root,"XTTS-v2","config.json"))
class XTTS(CustomerLLM):
    
    model_path: str = Field(None, alias='model_path')
    processor: Any = None
    file_path: str = "./"
    sample_rate: Any = 24000
    save_to_file: bool = False
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    

    def __init__(self, model_path: str = os.path.join(model_root,"XTTS-v2"),**kwargs):
        super(XTTS, self).__init__(llm=Xtts.init_from_config(config))
        self.model_path = model_path
        self.model.load_checkpoint(config, checkpoint_dir=self.model_path, eval=True)
        self.model.cuda()

    @property
    def _llm_type(self) -> str:
        return "coqui/XTTS-v2"
    
    @property
    def model_name(self) -> str:
        return "text2speech"
    
    def _call(
        self,
        prompt: Union[str,any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        generate_speech = kwargs.pop("language","zh-cn")
        outputs = self.model.synthesize(
            prompt,
            config,
            speaker_wav=os.path.join(self.model_path,"samples/zh-cn-sample.wav"),
            gpt_cond_len=3,
            language=generate_speech,
        )
        # print(outputs["wav"])
        sd.play(outputs["wav"],self.sample_rate, blocking=False)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
# if __name__ == '__main__':
#     sd = Whisper()
#     sd.predict("Hello, my dog is cute")