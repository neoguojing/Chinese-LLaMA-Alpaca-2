# 定义目标文件
PWD = $(shell pwd)
HFModelDIR = $(PWD)/model/llama/llama-2-7b/hf
LORAModelDIR = $(PWD)

LORA = 0
USE_CPU = --only_cpu
# 推理
inference:	
	echo $(HFModelDIR)
	
		echo $(HFModelDIR)
		python scripts/inference/inference_hf.py \
			--base_model $(HFModelDIR) \
			$(USE_CPU) \
			--with_prompt \
			--interactive
	else
		python scripts/inference/inference_hf.py \
			--base_model $(HFModelDIR) \
			--lora_model $(LORAModelDIR) \
			$(USE_CPU) \
			--with_prompt \
			--interactive

# # 训练
train:
	scripts/training/run_pt.sh

chat:
	scripts/training/run_sft.sh

init:
	pip install -r requirements.txt

# 默认规则
.PHONY: inference train chat
