
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from diffusers import DiffusionPipeline,DPMSolverMultistepScheduler
from diffusers import LCMScheduler, AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
    CallbackManagerForLLMRun
)
from pydantic import  Field
from apps.base import Task,CustomerLLM
from apps.config import model_root
import hashlib



def calculate_md5(string):
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode('utf-8'))
    md5_value = md5_hash.hexdigest()
    return md5_value

class StableDiff(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    tokenizer: Any = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = "./"

    # def __init__(self, model_path: str=os.path.join(model_root,"stable-diffusion"),**kwargs):
    #     super(StableDiff, self).__init__(
    #         llm=DiffusionPipeline.from_pretrained(
    #             model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    #     ))
    #     self.model_path = model_path
    #     # self.model = DiffusionPipeline.from_pretrained(
    #     #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    #     #     cache_dir=os.path.join(model_root,"stable-diffusion")
    #     # )
    #     # self.model.save_pretrained(os.path.join(model_root,"stable-diffusion"))
        
    #     # self.model = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")
    #     # self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)
    #     # self.model.enable_attention_slicing()
    #     # 推理速度变慢
    #     # self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
    #     # self.model.to(self.device)
    #     # 使用cpu和to('cuda')互斥，内存减小一半
    #     self.model.enable_model_cpu_offload()
    #     # 加速
    #     # adapter_id = "latent-consistency/lcm-lora-sdxl"
    #     # self.model.load_lora_weights(adapter_id)
    #     # self.model.fuse_lora()
    #     # self.model.save_lora_weights(os.path.join(model_root,"stable-diffusion"),unet_lora_layers)

    def __init__(self, model_path: str=os.path.join(model_root,"stable-diffusion"),**kwargs):
        super(StableDiff, self).__init__(
            llm=AutoPipelineForText2Image.from_pretrained(
                os.path.join(model_root,"sdxl-turbo"), torch_dtype=torch.float16, variant="fp16"
        ))
        self.model_path = model_path
        self.model.to(self.device)
        # 使用cpu和to('cuda')互斥，内存减小一半
        self.model.enable_model_cpu_offload()


    @property
    def _llm_type(self) -> str:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    
    @property
    def model_name(self) -> str:
        return "text2image"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        # image = self.model(
        #     **self.get_inputs(prompt)
        # ).images[0]

        image = self.model(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        file_name = calculate_md5(prompt)+".png"
        path = os.path.join(self.file_path, file_name)
        image.save(path)

        return path

    def get_inputs(self,prompt:str,batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = batch_size * [prompt]

        return {"prompt": prompts, "generator": generator, "num_inference_steps": self.n_steps}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    

class Image2Image(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    refiner: Any = None
    tokenizer: Any = None
    n_steps: int = 20
    high_noise_frac: float = 0.8
    file_path: str = "./"

    def __init__(self, model_path: str=os.path.join(model_root,"sdxl-turbo"),**kwargs):
        super(Image2Image, self).__init__(
            llm=AutoPipelineForImage2Image.from_pretrained(
                os.path.join(model_root,"sdxl-turbo"), torch_dtype=torch.float16, variant="fp16"
        ))
        # self.model.save_pretrained(os.path.join(model_root,"sdxl-turbo"))
        self.model_path = model_path

    @property
    def _llm_type(self) -> str:
        return "stabilityai/sdxl-turbo"
    
    @property
    def model_name(self) -> str:
        return "image2image"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        output = ""
        image_path = kwargs.pop("image_path","")
        if image_path != "":
            init_image = load_image(image_path).resize((512, 512))
            image = self.model(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
            file_name = calculate_md5(prompt)+".png"
            output = os.path.join(self.file_path, file_name)
            image.save(output)
        return output

    def get_inputs(self,prompt:str,batch_size=1):
        generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        prompts = batch_size * [prompt]

        return {"prompt": prompts, "generator": generator, "num_inference_steps": self.n_steps}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}




# if __name__ == '__main__':
#     sd = Image2Image()
#     sd.predict("a strong man")