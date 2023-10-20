from langchain.llms import OpenAI
from langchain import HuggingFacePipeline
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = 0
os.environ['OPENAI_API_KEY'] = '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'
        
class ModelFactory:
    @staticmethod
    def get_model(model_name,model_path):
        if model_name == "openai":
            return OpenAI()
        elif model_name == "qwen": 
            load_type=torch.bfloat16
            return HuggingFacePipeline.from_model_id(model_id=model_path,
                task="text-generation",
                device=0,
                model_kwargs={
                            "torch_dtype" : load_type,
                            "low_cpu_mem_usage" : True,
                            "temperature": 0.2,
                            "max_length": 1000,
                            "repetition_penalty":1.1}
                )
        elif model_name == "llama": 
            load_type=torch.float16
            return HuggingFacePipeline.from_model_id(model_id=model_path,
                task="text-generation",
                device=0,
                model_kwargs={
                            "torch_dtype" : load_type,
                            "low_cpu_mem_usage" : True,
                            "temperature": 0.2,
                            "max_length": 1000,
                            "repetition_penalty":1.1}
                )
        else:
            raise Exception("Invalid model name")