
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from diffusers import DiffusionPipeline
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

def calculate_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode('utf-8'))
    md5_value = md5_hash.hexdigest()
    return md5_value

class StableDiff(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    model: Any = None 
    refiner: Any = None
    tokenizer: Any = None
    n_steps: int = 40
    high_noise_frac: float = 0.8
    file_path: str = "./"

    def __init__(self, model_path: str=os.path.join(model_root,"stable-diffusion"),**kwargs):
        super(StableDiff, self).__init__()
        self.model_path = model_path
        # self.model = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        #     cache_dir=os.path.join(model_root,"stable-diffusion")
        # )
        # self.model.save_pretrained(os.path.join(model_root,"stable-diffusion"))
        self.model = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        )
        self.model.to(self.device)
        # self.refiner = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-refiner-1.0",
        #     text_encoder_2=self.model.text_encoder_2,
        #     vae=self.model.vae,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        #     cache_dir=os.path.join(model_root,"stable-diffusion")
        # )
        # self.refiner.to(self.device)

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        

    @property
    def _llm_type(self) -> str:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        image = self.model(
            prompt=prompt,
            # num_inference_steps=self.n_steps,
            # denoising_end=self.high_noise_frac,
            # output_type="latent",
        ).images[0]
        # image = self.refiner(
        #     prompt=prompt,
        #     num_inference_steps=self.n_steps,
        #     denoising_start=self.high_noise_frac,
        #     image=image,
        # ).images[0]

        file_name = calculate_md5(prompt)+".png"
        path = os.path.join(self.file_path, file_name)
        image.save(path)

        return path

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

class Text2Image(BaseTool):
  name = "Text to Image"
  description = "Text to image diffusion model capable of generating photo-realistic images given any text input."
  model = StableDiff()
  def _run(self,input:str):
      return self.model.predict(input)

if __name__ == '__main__':
    sd = StableDiff()
    sd.predict("a beauty chinese girl")