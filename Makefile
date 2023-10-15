 # Define target files
PWD = $(shell pwd)
MODEL_DIR := $(PWD)/model

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
FIRST_INSTRUCTION="hello"

# data
GENE_DATA_DIR := $(PWD)/dataset/generate
CHAT_DATA_DIR := $(PWD)/dataset/chat
CHAT_VALIDATE_FILE := $(PWD)/dataset/chat/validate
CACEH_DATA_DIR := $(PWD)/cache

# llama
HFModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
HFTokenModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
HFChatModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf
HFChatTokenModelDIR := $(MODEL_DIR)/llama/llama-2-7b/hf


# chinise llama
ZHModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf
ZHTokenModelDIR := $(MODEL_DIR)/chinese/chinese-llama-2-7b-hf/
ZHChatModelDIR := $(MODEL_DIR)/chinese/chinese-alpaca-2-7b-hf
ZHChatTokenDIR := $(MODEL_DIR)/chinese/chinese-alpaca-2-7b-hf/

# qwen
QWenModelDIR := $(MODEL_DIR)/chinese/Qwen-7B/
QWenTokenModelDIR := $(MODEL_DIR)/chinese/Qwen-7B/
QWenChatModelDIR := $(MODEL_DIR)/chinese/Qwen-7B-Chat/
QWenChatTokenModelDIR := $(MODEL_DIR)/chinese/Qwen-7B-Chat/

# lora
LORAModelDIR := $(MODEL_DIR)/pt_lora_model


# output
ModelOutputDIR := $(MODEL_DIR)
MergeTargetDIR := $(MODEL_DIR)/target

# eval
EvalPath := $(PWD)/eval



cpu := --only_cpu
isLLaMaLikeModel := --llama
isChat := --chat

export cuda?=1
export chat?=1
export qwen?=1
export test?=0
export zh?=0
export llama?=0

BUILD_FLAGS:=LLAMA_OPENBLAS=1

BaseModelPath := $(ZHModelDIR)
BaseTokenPath := $(ZHTokenModelDIR)

ifeq ($(cuda), 1)
	cpu :=
	BUILD_FLAGS:=LLAMA_CUBLAS=1
endif

ifeq ($(chat), 0)

	ifeq ($(qwen), 1)
		BaseModelPath :=$(QWenModelDIR)
		BaseTokenPath := $(QWenTokenModelDIR)
		isLLaMaLikeModel :=
		isChat :=
	endif

	ifeq ($(zh), 1)
		BaseModelPath :=$(ZHModelDIR)
		BaseTokenPath := $(ZHTokenModelDIR)
		isChat :=
	endif

	ifeq ($(llama), 1)
		BaseModelPath :=$(HFModelDIR)
		BaseTokenPath := $(HFTokenModelDIR)
		isChat :=
	endif

else

	ifeq ($(qwen), 1)
		BaseModelPath :=$(QWenChatModelDIR)
		BaseTokenPath := $(QWenChatTokenModelDIR)
		isLLaMaLikeModel :=
	endif

	ifeq ($(zh), 1)
		BaseModelPath :=$(ZHChatModelDIR)
		BaseTokenPath := $(ZHChatTokenDIR)
	endif

	ifeq ($(llama), 1)
		BaseModelPath :=$(HFChatModelDIR)
		BaseTokenPath := $(HFChatTokenModelDIR)
	endif

endif


 
 # Inference
run:
	@echo "Using model path: $(BaseModelPath)"
	python scripts/inference/inference.py \
		--base_model $(BaseModelPath) \
		--tokenizer_path $(BaseTokenPath) \
		$(cpu) \
		$(isLLaMaLikeModel) \
		$(isChat)


lora:
	@echo "Using model path: $(BaseModelPath)"
	python scripts/inference/inference_hf.py \
				--base_model $(BaseModelPath) \
				--lora_model $(LORAModelDIR) \
				$(cpu) \
				$(isLLaMaLikeModel) \
				$(isChat) \
				--with_prompt
 # Training
train:
	# rm -rf cache/*
	cd scripts/training && ./run_pt.sh $(BaseModelPath) $(BaseTokenPath) $(GENE_DATA_DIR) $(CACEH_DATA_DIR) $(ModelOutputDIR) 

sft:
	# rm -rf cache/*
	cd scripts/training && run_sft.sh $(BaseModelPath) $(BaseTokenPath) $(CHAT_DATA_DIR) $(ModelOutputDIR) $(CHAT_VALIDATE_FILE)

llama.cpp:
	cd llama.cpp && make $(BUILD_FLAGS)

quantize:
	cd llama.cpp && python convert.py $(BaseModelPath)
	cd llama.cpp && ./quantize $(BaseModelPath)/ggml-model-f16.gguf $(BaseModelPath)/ggml-model-q4_0.gguf q4_0

test:
	cd llama.cpp && ./main -m "$(BaseModelPath)/ggml-model-q4_0.gguf" \
	--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
	--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]'

deploy:
	cd ../text-generation-webui && python server.py --model-dir $(BaseModelPath) --loader llamacpp --model $(BaseModelPath)/ggml-model-q4_0.gguf

dataset:
	rm -rf cache
	python scripts/training/dataset_handler.py \
    --tokenizer_name_or_path $(BaseTokenPath) \
    --dataset_dir $(GENE_DATA_DIR) \
    --data_cache_dir $(CACEH_DATA_DIR) \
    --block_size 512

merge:
	python scripts/merge_llama2_with_chinese_lora_low_mem.py \
    --base_model $(BaseModelPath) \
    --lora_model $(LORAModelDIR) \
    --output_type huggingface \
    --output_dir $(MergeTargetDIR)

ceval:
	cd scripts/ceval && python eval.py \
		--model_path ${BaseModelPath} \
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
