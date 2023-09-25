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

LORAModelDIR := $(MODEL_DIR)/pt_lora_model

ChatPreTrainModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf
ChatPreTrainTokenDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf

ModelOutputDIR := $(MODEL_DIR)
TrainTargetDIR := $(MODEL_DIR)/target

ModelPath := $(ChatModelDIR)
EvalPath := $(PWD)/eval

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
FIRST_INSTRUCTION="hello"

cpu := --only_cpu

export cuda?=1

BUILD_FLAGS:=LLAMA_OPENBLAS=1

ifeq ($(cuda), 1)
	cpu :=
	BUILD_FLAGS:=LLAMA_CUBLAS=1
endif


ifeq ($(chat), 0)
	ModelPath :=$(TrainTargetDIR) 
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
	# rm -rf cache/*
	cd scripts/training && ./run_pt.sh $(ZHModelDIR) $(ZHTOkenModelDIR) $(GENE_DATA_DIR) $(CACEH_DATA_DIR) $(ModelOutputDIR) 

sft:
	# rm -rf cache/*
	cd scripts/training && run_sft.sh $(ChatPreTrainModelDIR) $(ChatPreTrainTokenDIR) $(CHAT_DATA_DIR) $(ModelOutputDIR) $(CHAT_VALIDATE_FILE)

llama.cpp:
	cd llama.cpp && make $(BUILD_FLAGS)

quantize:
	cd llama.cpp && python convert.py $(ChatModelDIR)
	cd llama.cpp && ./quantize $(ChatModelDIR)/ggml-model-f16.gguf $(ChatModelDIR)/ggml-model-q4_0.gguf q4_0

test:
	cd llama.cpp && ./main -m "$(ChatModelDIR)/ggml-model-q4_0.gguf" \
	--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
	--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]'

deploy:
	cd ../text-generation-webui && python server.py --model-dir $(ChatModelDIR) --loader llamacpp --model $(ChatModelDIR)/ggml-model-q4_0.gguf

dataset:
	rm -rf cache
	python scripts/training/dataset_handler.py \
    --tokenizer_name_or_path $(ZHTOkenModelDIR) \
    --dataset_dir $(GENE_DATA_DIR) \
    --data_cache_dir $(CACEH_DATA_DIR) \
    --block_size 512

merge:
	python scripts/merge_llama2_with_chinese_lora_low_mem.py \
    --base_model $(ZHModelDIR) \
    --lora_model $(LORAModelDIR) \
    --output_type huggingface \
    --output_dir $(TrainTargetDIR)

ceval:
	cd scripts/ceval && python eval.py \
		--model_path ${ZHModelDIR} \
		--cot False \
		--few_shot False \
		--with_prompt False \
		--constrained_decoding True \
		--temperature 0.2 \
		--n_times 1 \
		--ntrain 5 \
		--do_save_csv False \
		--do_test False \
		--output_dir ${EvalPath}

init:
	conda create -n train python=3.10
	pip install -r requirements.txt
	# git clone https://github.com/ggerganov/llama.cpp
 
 # Default rules
.PHONY: run train init prepare deploy quantize test dataset lora ceval
