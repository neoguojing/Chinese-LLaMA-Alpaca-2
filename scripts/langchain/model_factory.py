from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
import torch
import os
from transformers import GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['OPENAI_API_KEY'] = ''

generation_config = GenerationConfig(
        temperature=0.2, #控制采样的温度参数。较高的温度值会增加生成的随机性，较低的温度值会增加生成的确定性。通常在 (0, 1] 范围内设置
        top_k=40, #表示在采样时保留概率最高的 k 个标记
        top_p=0.9, #表示在采样时的重复惩罚（repetition penalty）。较高的值会鼓励模型生成更加多样化的文本，较低的值会鼓励模型生成更加一致的文本
        do_sample=True, #如果设置为 True，则生成时将使用采样策略，从模型的概率分布中随机选择下一个标记。如果设置为 False，则将使用贪婪（greedy）策略选择概率最高的标记。
        num_beams=1, # 较大的 num_beams 值会增加生成的多样性，因为更多的候选项被保留并参与下一步的扩展。然而，较大的值也会增加计算开销
        repetition_penalty=1.1, #同top_p
        max_new_tokens=400 # 参数用于限制生成的文本中新增标记的数量。标记可以是单词、子词或字符，具体取决于所使用的模型和分词器，参数是相对于输入上下文而言的。如果输入上下文已经包含一些标记，那么生成的文本中新增标记的数量将计算为生成文本和输入上下文之间的差异
    )

class ModelFactory:
    temperature = 0

    @staticmethod
    def get_model(model_name,model_path=""):
        if model_name == "openai":
            return OpenAI()
        elif model_name == "qwen": 
            model_path = "../../model/chinese/Qwen-7B-Chat"
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
            model_path = "../../model/chinese/chinese-alpaca-2-7b-hf"
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