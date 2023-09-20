import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict,prepare_model_for_int8_training
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
)
import logging
logger = logging.getLogger(__name__)
import os


device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    flash_attn : Optional[bool] = field(default=False)
    use_cache: Optional[bool] = field(default=False)
    quantization: Optional[bool] = field(default=False)

    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


# 加载预训练模型
def load_pretrained_model(model_args):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # 为了节约时间空间换时间 gradient_checkpointing和use_cache不能同时设置为True
    config.use_cache = model_args.use_cache

    if model_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # torch_dtype变量用于指定PyTorch张量的数据类型
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),  # 指示模型是否来自TensorFlow的检查点（.ckpt文件）。如果模型路径包含.ckpt，则为True；否则为False
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,  # 指示是否启用低CPU内存使用模式
        load_in_8bit=True if model_args.quantization else None,  # 指示是否以8位精度加载模型。这通常用于量化（quantization）训练
        device_map="auto" if model_args.quantization else device_map,
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
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
         # 可训练参数通常是指神经网络中的权重（weights）和偏置（biases
        model.print_trainable_parameters()
        # state_dict字典将每个模型参数的名称作为键（key），将对应参数的张量值作为值（value）。
        # 这些参数张量包含了模型在训练过程中学习到的权重和偏置等可训练参数。
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

        if model_args.quantization:
            model = prepare_model_for_int8_training(model)
    return model

   