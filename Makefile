 # Define target files
PWD = $(shell pwd)
MODEL_DIR := $(PWD)/model
GENE_DATA_DIR := $(PWD)/dataset/generate
CHAT_DATA_DIR := $(PWD)/dataset/chat
CACEH_DATA_DIR := $(PWD)/cache

HFModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
HFTOkenModelDIR := $(MODEL_DIR)/llama/llama-2-7b/tokenizer.model
ChatModelDIR := $(MODEL_DIR)/chinese/chinese-alpaca-2-7b-hf
LORAModelDIR := $(MODEL_DIR)/lora/lora-2-7b/hf
ModelOutputDIR := $(MODEL_DIR)

ModelPath := $(ChatModelDIR)

cpu := --only_cpu
 

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
	cd scripts/training
	run_pt.sh $(HFModelDIR) $(HFTOkenModelDIR) $(GENE_DATA_DIR) $(CACEH_DATA_DIR) $(ModelOutputDIR)

sft:
	scripts/training/run_sft.sh

 
init:
	pip install -r requirements.txt
 
 # Default rules
.PHONY: run train init
