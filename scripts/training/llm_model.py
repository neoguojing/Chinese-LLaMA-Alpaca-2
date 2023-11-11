import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict,prepare_model_for_kbit_training
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    GPTQConfig,
    BitsAndBytesConfig,
    deepspeed
)
import logging
logger = logging.getLogger(__name__)
import os



max_memory = {0: "23GiB", 1: "23GiB", "cpu": "32GiB"} 
device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper

@singleton
class TokenizerSingleton:
    def __init__(self, tokenizer_name_or_path:str,model_max_length:int=8192,llama: bool=False,):
        if llama:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_name_or_path,
                use_fast=False,
                trust_remote_code=True)
            # 强制添加句子结束标志
            self.tokenizer.add_eos_token = True  
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=False,
                trust_remote_code=True)
            # pad标记等同于文档结束标志
            self.tokenizer.pad_token_id = self.tokenizer.eod_id

def create_tokenizer(tokenizer_name_or_path:str,model_max_length:int=None,llama: bool=False):
    tokenizer = TokenizerSingleton(tokenizer_name_or_path,model_max_length,llama).tokenizer
    return tokenizer

def get_model_config(model_args):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,trust_remote_code=True)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True,)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    # 为了节约时间空间换时间 gradient_checkpointing和use_cache不能同时设置为True
    config.use_cache = model_args.use_cache
    return config

def get_quantization_config(model_args):
    quantization_config = None
    if model_args.llama:
        quantization_config = None

    elif model_args.use_lora and model_args.q_lora:
        quantization_config = GPTQConfig(
            bits=4, disable_exllama=True
        )
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type='nf4'
    #     )
    

    return quantization_config

# 加载预训练模型
def load_pretrained_model(model_args):

    config = get_model_config(model_args)
    # TODO
    if model_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # torch_dtype变量用于指定PyTorch张量的数据类型
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if model_args.llama:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),  # 指示模型是否来自TensorFlow的检查点（.ckpt文件）。如果模型路径包含.ckpt，则为True；否则为False
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            max_memory = max_memory,
            quantization_config = get_quantization_config(model_args),
            low_cpu_mem_usage=True,  # 指示是否启用低CPU内存使用模式
            device_map="auto" if device_map == None else device_map,
            trust_remote_code=True,
        )
    else:
        # INT8 + DeepSpeed also isn't supported
        # deepSpeed isn't can't be used when using device_map or low_cpu_mem_usage
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),  # 指示模型是否来自TensorFlow的检查点（.ckpt文件）。如果模型路径包含.ckpt，则为True；否则为False
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            max_memory = max_memory,
            quantization_config = get_quantization_config(model_args),
            low_cpu_mem_usage=True if model_args.use_lora and not model_args.q_lora else False,
            device_map="auto" if model_args.use_lora and not model_args.q_lora else None,
            trust_remote_code=True,
        )
    return model

def determine_vocab_size(model, tokenizer_vocab_size):
   
    # 获取模型的输出嵌入层
    output_embeddings = model.get_output_embeddings()
    # 获取输出嵌入层的权重矩阵
    weights = output_embeddings.weight
    # 获取权重矩阵的第一个维度的大小，即词汇表的大小
    model_vocab_size = weights.size(0)  
    # tokenizer_vocab_size = len(tokenizer)
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    if tokenizer_vocab_size!= 55296:
        raise ValueError(f"The vocab size of tokenizer is {tokenizer_vocab_size}, not 55296. Please use Chinese-LLaMA-2 tokenizer.")
    if model_vocab_size!= tokenizer_vocab_size:
        print(f"Resize model vocab size to {tokenizer_vocab_size}")
        # 调整嵌入层的大小以匹配词汇表大小
        model.resize_token_embeddings(tokenizer_vocab_size)
    return model


def create_peft_model(model, model_args):
    """
    Create a PEFT model based on the given model and training arguments.

    Args:
        model (LlamaForCausalLM): The pre-trained model to be used as a starting point for PEFT.
        training_args (TrainingArguments): The training arguments containing information about the training process.

    Returns:
        A PEFT model based on the given model and training arguments.
    """
    if model_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, model_args.peft_path, device_map=device_map)
    else:
        logger.info("Init new peft model")
        target_modules = model_args.trainable.split(',')
        modules_to_save = model_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        
        lora_rank = model_args.lora_rank
        lora_dropout = model_args.lora_dropout
        lora_alpha = model_args.lora_alpha
        lora_bias = model_args.lora_bias
        print(f"target_modules: {target_modules}")
        print(f"lora_rank: {lora_rank}")
        print(f"modules_to_save: {modules_to_save}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=modules_to_save)
        
        if model_args.q_lora:
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, peft_config)

        

         # 可训练参数通常是指神经网络中的权重（weights）和偏置（biases
        model.print_trainable_parameters()
        # state_dict字典将每个模型参数的名称作为键（key），将对应参数的张量值作为值（value）。
        # 这些参数张量包含了模型在训练过程中学习到的权重和偏置等可训练参数。
        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        # ).__get__(model, type(model))

    return model

def print_number_of_trainable_model_parameters(model):
  trainable_model_params = 0
  all_model_params = 0
  for _, param in model.named_parameters():
    all_model_params += param.numel()
    if param.requires_grad:
      trainable_model_params += param.numel()
  print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
  return trainable_model_params