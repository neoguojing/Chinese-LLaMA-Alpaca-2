
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from transformers.utils.versions import require_version

from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "数据目录"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "远程数据名称"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "训练时输入的块大小，默认为模型支持的最大序列长度"
            )
        },
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "训练集和测试集的拆分比例"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "数据预处理需要的线程数"},
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "数据缓存目录"})

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    flash_attn : Optional[bool] = field(default=False)
    use_cache: Optional[bool] = field(default=False)
    quantization: Optional[bool] = field(default=False)

    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    lora_bias: Optional[str] = "none"
    
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)

    q_lora: bool = field(
        default=True,
        metadata={
            "help": (
                "use qlora for peft or quantize."
            )
        },
    )

    gptq : bool = field(
        default=True,
        metadata={
            "help": (
                "use gptq for quantize. Prier to qlora"
            )
        },
    )

    llama: bool = field(
        default=False,
        metadata={
            "help": (
                "llama based model.default False"
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


# @dataclass
# class LLamaConfig(DataTrainingArguments,ModelArguments,TrainingArguments):
    
# @dataclass
# class LLamaChatConfig(DataTrainingArguments,ModelArguments,TrainingArguments):

# @dataclass
# class QwenChatConfig(DataTrainingArguments,ModelArguments,TrainingArguments):

# @dataclass
# class ConfigFactory:
#     """配置工厂"""
#     configs: Dict[str, Type[BaseConfig]] = field(default_factory=dict)

#     def create(self, name: str) -> BaseConfig:
#         config_cls = self.configs[name]
#         return config_cls()