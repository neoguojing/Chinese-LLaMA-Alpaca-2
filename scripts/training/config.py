
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from transformers.utils.versions import require_version
import os

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
        default=512,
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
        default=4,
        metadata={"help": "数据预处理需要的线程数"},
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "数据缓存目录"})

    llm_type: Optional[str] = field(
        default="qwen", metadata={"help": "模型类别"}
    )

    chat: Optional[bool] = field(
        default=True, metadata={"help": "是否聊天模型"}
    )

    def __post_init__(self):
        if self.data_cache_dir == None:
            self.data_cache_dir = os.path.join(self.dataset_dir, "cache")

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
        default=False,
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
        metadata={"help": "模型下载的缓存目录"},
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

    use_lora: bool = field(
        default=True,
        metadata={
            "help": (
                "use lora for peft or quantize."
            )
        },
    )

    q_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "use qlora for peft or quantize."
            )
        },
    )

    gptq : bool = field(
        default=False,
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
                "config_overrides can't be used in combination with config_name or model_name_or_path"
            )
        
        if self.llama:
            self.torch_dtype = "float16"

        if self.tokenizer_name_or_path == None:
            self.tokenizer_name_or_path = self.model_name_or_path
        

LLama_Config = {
    "model_name_or_path": "",  # 模型名称或路径
    "tokenizer_name_or_path": "",  # 分词器名称或路径
    "llama": True,  # 是否使用 LLama 模型
    "torch_dtype": "float16",  # 使用的 Torch 数据类型
    "lora_rank": 64,  # 更新矩阵的秩，以整数表示。秩越低，更新矩阵规模越小，可训练参数越少
    "lora_alpha": 128,  # 低秩近似缩放因子，缩放因子值越大，重构后的矩阵与原始矩阵越相似；缩放因子值越小，参数量会更小，但重构误差也会更大
    "trainable": "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",  # 可更新的参数层
    "lora_dropout": 0.05,  # 不存在？
    "modules_to_save": "embed_tokens,lm_head",  # 除LoRa层外需要设置为可训练并保存到最终检查点的模块列表。这些模块通常包括模型自定义的头层，用于初始化下游调优任务。

    "dataset_dir": "",  # 数据集目录
    "data_cache_dir": "",  # 数据缓存目录
    "validation_split_percentage": 0.001,  # 验证集百分比
    "preprocessing_num_workers": 8,  # 预处理工作线程数
    "block_size": 512,  # 块大小
}


LLama_Train_Config = {
    "dataloader_num_workers": 4,  # 数据加载时使用的子进程数
    "per_device_train_batch_size": 1,  # 它控制了分布式训练场景下，每个独立设备处理的批次样本数量
    "do_train": True,  # 开启训练模式
    "do_eval": False,  # 开启评估
    "num_train_epochs": 1,  # 总训练周期数
    "gradient_accumulation_steps": 8,  # 在反向更新参数时，累积多少步梯度

    "lr_scheduler_type": "cosine",  # 学习率衰减策略，学习率随训练进度进行余弦下降
    "learning_rate": 2e-4,  # AdamW的学习率
    "warmup_ratio": 0.05,  # 0.1，则总训练步骤的前10%时间内，学习率从0逐步线性增长到最大学习率，在深度学习模型训练开始时，直接使用最大学习率可能导致模型卡住或发散
    "weight_decay": 0.01,  # AdamW优化器中，应用于除去Bias和LayerNorm参数以外的所有层的权重衰减值（如果不为零）

    "logging_strategy": "steps",  # 日志策略
    "logging_steps": 10,  # 日志更新的步数
    "logging_first_step": True,  # 是否记录第一步

    "output_dir": "",  # 模型checkpoint输出目录
    "overwrite_output_dir": True,  # 覆盖输出目录
    "save_strategy": "steps",  # 检查点保存策略
    "save_total_limit": 3,  # 限制的检查点个数
    "save_steps": 200,  # 两个检查点之间的间隔步数

    "fp16": True,  # 使用半精度浮点数

    "deepspeed":"ds_config_zero2.json",
    # "gradient_checkpointing": True,  # 梯度检查点技术，可以减少内存开销，但代价是反向传播会较慢
    # "flash_attn": True,
}


LLama_Chat_Config = {
    "model_name_or_path": "",
    "tokenizer_name_or_path": "",
    "lora_rank": 64,
    "lora_alpha": 128,
    "trainable": "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
    "lora_dropout": 0.05,
    "modules_to_save": "embed_tokens,lm_head",
    "torch_dtype": "float16",
    "llama": True,

    "dataset_dir": "",
    "preprocessing_num_workers": 8,
    "block_size": 512,
}

LLama_Chat_Train_Config = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "do_train": True,
    "do_eval": True,
    "fp16": True,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "learning_rate": 1e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0,
    "logging_strategy": "steps",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_total_limit": 3,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 200,
    "gradient_accumulation_steps": 8,
    "output_dir": "",
    "overwrite_output_dir": True,
    "logging_first_step": True,

    "deepspeed": "ds_config_zero2.json",
}

# qlora 与 deepspeed3 不兼容
Qwen_Chat_Config = {
    "model_name_or_path": "",
    "lora_rank": 64,
    "lora_alpha": 16,
    "trainable": "c_attn,c_proj,w1,w2",
    "lora_dropout": 0.05,
    "modules_to_save": "wte,lm_head",
    "dataset_dir": "",
    "block_size": 512,
    "use_lora": True,
    "q_lora": False,
}

Qwen_Chat_Train_Config = {
    "output_dir": "",
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "evaluation_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 10,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "adam_beta2": 0.95,
    "warmup_ratio": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "report_to": "none",
    "gradient_checkpointing": True,
    "do_train": True,
    "do_eval": True,
    # "bf16": True,
    "deepspeed": "ds_config_zero2.json",
    "fp16": True
}


@dataclass
class LLMConfig(DataTrainingArguments,ModelArguments):
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "config_overrides can't be used in combination with config_name or model_name_or_path"
            )
        if self.data_cache_dir is None:
            self.data_cache_dir = os.path.join(self.dataset_dir, "cache")

        if self.llama:
            self.torch_dtype = "float16"

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.q_lora or 'chat' in self.model_name_or_path.lower():
            self.modules_to_save = None

        if self.use_lora and not self.q_lora:
            self.torch_dtype = "bfloat16"
            # self.deepspeed = None
            # self.bf16 = True
            # self.fp16 = False
        elif self.use_lora and self.q_lora:
            self.torch_dtype = "float16"
            # self.fp16 = True
            # self.bf16 = False

@dataclass
class ConfigFactory(DataTrainingArguments,ModelArguments,TrainingArguments):
 
    def create(self):
          if self.chat:
              if self.llm_type == "qwen":
                  Qwen_Chat_Config["model_name_or_path"] = self.model_name_or_path
                  Qwen_Chat_Config["tokenizer_name_or_path"] = self.tokenizer_name_or_path
                  Qwen_Chat_Config["dataset_dir"] = self.dataset_dir
  
                  Qwen_Chat_Train_Config["output_dir"] = self.output_dir
                  return LLMConfig(**Qwen_Chat_Config),TrainingArguments(**Qwen_Chat_Train_Config)
              if self.llm_type == "llama":
                  LLama_Chat_Config["model_name_or_path"] = self.model_name_or_path
                  LLama_Chat_Config["tokenizer_name_or_path"] = self.model_name_or_path
                  LLama_Chat_Config["dataset_dir"] = self.dataset_dir
                  LLama_Chat_Train_Config["output_dir"] = self.output_dir
                  return LLMConfig(**LLama_Chat_Config),TrainingArguments(**LLama_Chat_Train_Config)
          else:
              LLama_Config["model_name_or_path"] = self.model_name_or_path
              LLama_Config["tokenizer_name_or_path"] = self.model_name_or_path
              LLama_Config["dataset_dir"] = self.dataset_dir
              LLama_Train_Config["output_dir"] = self.output_dir
              return LLMConfig(**LLama_Config),TrainingArguments(**LLama_Train_Config)
  
    
