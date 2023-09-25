# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh)
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh) carefully before running the script
lr=2e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=$1
chinese_tokenizer_path=$2
dataset_dir=$3
data_cache=$4
per_device_train_batch_size=1
gradient_accumulation_steps=8
block_size=512
output_dir=$5

nproc_per_node=1

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node ${nproc_per_node} llm_fine_tune.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --fp16=True \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --deepspeed ${deepspeed_config_file}
    # --gradient_checkpointing True \
    # --modules_to_save ${modules_to_save} \
    # --optim adamw_bnb_8bit \
    # --flash_attn True \
    # --quantization True \
    # --optim adafactor
    # --bf16 False
    # --torch_dtype auto
    
