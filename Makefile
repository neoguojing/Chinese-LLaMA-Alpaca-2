# 定义目标文件
PWD = $(shell pwd)
HFModelDIR = $(PWD)/model/llama/llama-2-7b/hf
ChatModelDIR = $(PWD)model/chinese/chinese-alpaca-2-7b-hf

ModelPath = $(ChatModelDIR)

LORAModelDIR = $(PWD)

lora = 0
cpu = --only_cpu

chat = 1
ifeq ($(chat), 0)
	ModelPath = $(HFModelDIR)
endif
# 推理
run:	
	echo $(ModelPath)
	ifeq ($(lora), 0)
		python scripts/inference/inference_hf.py \
			--base_model $(ModelPath) \
			$(cpu) \
			--with_prompt \
			--interactive
	else
		python scripts/inference/inference_hf.py \
			--base_model $(ModelPath) \
			--lora_model $(LORAModelDIR) \
			$(cpu) \
			--with_prompt \
			--interactive
	endif


# # 训练
train:
	ifeq ($(chat),0)
		scripts/training/run_pt.sh
	else 
		scripts/training/run_sft.sh
	endif

init:
	pip install -r requirements.txt

# 默认规则
.PHONY: inference train chat
