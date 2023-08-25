# 定义目标文件
HFModelDIR = my_project
LORAModelDIR = /



LORA := 0
# 推理
inference:	
	ifeq ($(LORA),0)
		python scripts/inference/inference_hf.py \
			--base_model $(HFModelDIR) \
			--with_prompt \
			--interactive
	else
		python scripts/inference/inference_hf.py \
			--base_model $(HFModelDIR) \
			--lora_model $(LORAModelDIR) \
			--with_prompt \
			--interactive

# # 训练
train:
	scripts/training/run_pt.sh

chat：
	run_clm_sft_with_peft.py



# 默认规则
.PHONY: inference train
