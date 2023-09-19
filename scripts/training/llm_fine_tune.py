from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
import sys
import os
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
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
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from dataset_handler import DataTrainingArguments
from llm_model import ModelArguments
from tokenizer import create_tokenizer,TokenizerArguments
@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    quantization: Optional[bool] = field(default=False)

# training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

# decay_parameters = get_parameter_names(model, [nn.LayerNorm])
# decay_parameters = [name for name in decay_parameters if "bias" not in name]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
#         "weight_decay": training_args.weight_decay,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
#         "weight_decay": 0.0,
#     },
# ]

# optimizer_kwargs = {
#     "betas": (training_args.adam_beta1, training_args.adam_beta2),
#     "eps": training_args.adam_epsilon,
# }
# optimizer_kwargs["lr"] = training_args.learning_rate
# adam_bnb_optim = bnb.optim.Adam8bit(
#     optimizer_grouped_parameters,
#     betas=(training_args.adam_beta1, training_args.adam_beta2),
#     eps=training_args.adam_epsilon,
#     lr=training_args.learning_rate,
# )


# training_args = TrainingArguments(
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     fp16=True,
#     optimizers=(adam_bnb_optim, None)
# )

# dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

# if training_args.gradient_checkpointing:
#     model.gradient_checkpointing_enable()

# accelerator = Accelerator(fp16=training_args.fp16)
# model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

# model.train()
# for step, batch in enumerate(dataloader, start=1):
#     loss = model(**batch).loss
#     loss = loss / training_args.gradient_accumulation_steps
#     accelerator.backward(loss)
#     if step % training_args.gradient_accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments,TokenizerArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args,token_arg = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args,token_arg = parser.parse_args_into_dataclasses()

    tokenizer = create_tokenizer(token_arg)