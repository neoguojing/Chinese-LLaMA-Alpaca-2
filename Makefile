 # Define target files
PWD = $(shell pwd)
MODEL_DIR := $(PWD)/model

HFModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
ChatModelDIR := $(MODEL_DIR)/chinese/chinese-alpaca-2-7b-hf
LORAModelDIR := $(MODEL_DIR)/lora/lora-2-7b/hf

ModelPath := $(ChatModelDIR)

lora := 0
cpu := --only_cpu
chat := 1
 

ifeq ($(chat), 0)
	ModelPath :=$(HFModelDIR) 
endif
 
 # Inference
run:
	@echo "Using model path: $(ModelPath)"
	python scripts/inference/inference_hf.py \
		--base_model $(ModelPath) \
		$(cpu) \
		--with_prompt \
		--interactive 


lora:
 	@echo "Using model path: $(ModelPath)"
	python scripts/inference/inference_hf.py \
				--base_model $(ModelPath) \
				--lora_model $(LORAModelDIR) \
				$(cpu) \
				--with_prompt \
				--interactive 

 # Training
train:
	scripts/training/run_pt.sh

sft:
	scripts/training/run_sft.sh

 
init:
	pip install -r requirements.txt
 
 # Default rules
.PHONY: run train init
