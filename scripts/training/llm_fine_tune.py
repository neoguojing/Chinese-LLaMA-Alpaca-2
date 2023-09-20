from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
import sys
import os
import math
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
from dataset_handler import DataTrainingArguments,create_tokenizer,TokenizerArguments,determine_block_size,preprocess_dataset
from llm_model import ModelArguments,load_pretrained_model,determine_vocab_size,create_peft_model
import logging
logger = logging.getLogger(__name__)
@dataclass
class MyTrainingArguments(TrainingArguments):
    
    debug_mode : Optional[bool] = field(default=False)

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
    block_size = determine_block_size(data_args,tokenizer)
    train_dataset,eval_dataset =  preprocess_dataset(data_args,block_size)
    model = load_pretrained_model(model_args)
    model = determine_vocab_size(model,len(tokenizer))
    model = create_peft_model(model,model_args)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator, #数据批处理和数据加载的数据整合器（data collator）对象。它处理训练和评估数据集的样本，并将它们整合成适当的格式供模型使用。
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None, #计算模型性能指标的函数
        preprocess_logits_for_metrics=preprocess_logits_for_metrics 
        if training_args.do_eval and not is_torch_tpu_available()
        else None, #对模型输出进行预处理以计算指标的函数
    )
    trainer.add_callback(SavePeftModelCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # 它用于评估模型对给定序列的预测能力和模型的概率分布的复杂度
        # 困惑度是对该概率分布的度量，表示模型对真实序列的预测能力。具体来说，困惑度越低表示模型对真实序列的预测越准确
        # 对于一个由N个词组成的序列，困惑度的计算如下：
        # 困惑度 = exp(交叉熵损失 / N)
        # 其中，交叉熵损失是模型对该序列的负对数似然（negative log-likelihood），N是序列的长度。
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)