 # Define target files
PWD = $(shell pwd)
MODEL_DIR := $(PWD)/model
GENE_DATA_DIR := $(PWD)/dataset/generate
CHAT_DATA_DIR := $(PWD)/dataset/chat
CHAT_VALIDATE_FILE := $(PWD)/dataset/chat/validate
CACEH_DATA_DIR := $(PWD)/cache

HFModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
HFTOkenModelDIR := $(MODEL_DIR)/llama/llama-2-7b/tokenizer.model

ZHModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf
ZHTOkenModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf/tokenizer.model

ChatModelDIR := $(MODEL_DIR)/chinese/chinese-alpaca-2-7b-hf

LORAModelDIR := $(MODEL_DIR)/lora/lora-2-7b/hf

ChatPreTrainModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf
ChatPreTrainTokenDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf



ModelOutputDIR := $(MODEL_DIR)

ModelPath := $(ChatModelDIR)

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
FIRST_INSTRUCTION="hello"

cpu := --only_cpu

export cuda?=0

BUILD_FLAGS:=LLAMA_OPENBLAS=1

ifeq ($(cuda), 1)
	cpu :=
	BUILD_FLAGS:=LLAMA_CUBLAS=1
endif


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
	rm -rf cache/*
	cd scripts/training && ./run_pt.sh $(ZHModelDIR) $(ZHTOkenModelDIR) $(GENE_DATA_DIR) $(CACEH_DATA_DIR) $(ModelOutputDIR)

sft:
	rm -rf cache/*
	cd scripts/training && run_sft.sh $(ChatPreTrainModelDIR) $(ChatPreTrainTokenDIR) $(CHAT_DATA_DIR) $(ModelOutputDIR) $(CHAT_VALIDATE_FILE)

llama.cpp:
	git clone https://github.com/ggerganov/llama.cpp
	cd llama.cpp && make $(BUILD_FLAGS)

quantize:
	cd llama.cpp && python convert.py $(ChatModelDIR)
	cd llama.cpp && ./quantize $(ChatModelDIR)/ggml-model-f16.gguf $(ChatModelDIR)/ggml-model-q4_0.gguf q4_0

test:
	cd llama.cpp && ./main -m "$(ChatModelDIR)/ggml-model-q4_0.gguf" \
	--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
	--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]'

deploy:


init: llama.cpp
	pip install -r requirements.txt
 
 # Default rules
.PHONY: run train init prepare deploy quantize test
