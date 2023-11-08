import argparse
import json, os
import torch
from typing import Tuple, List, Union, Iterable,Optional
from .generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)
import pdb
# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

history_max_len = 1000  # 模型记忆的最大token长度
history_token_ids = torch.tensor([[]], dtype=torch.long)

from transformers import LlamaForCausalLM, LlamaTokenizer,AutoConfig,AutoModelForCausalLM,AutoTokenizer
from transformers import GenerationConfig,PreTrainedTokenizer,PreTrainedModel
from transformers import BitsAndBytesConfig
from peft import  PeftModel

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

generation_config = GenerationConfig(
    temperature=0.2, #控制采样的温度参数。较高的温度值会增加生成的随机性，较低的温度值会增加生成的确定性。通常在 (0, 1] 范围内设置
    top_k=40, #表示在采样时保留概率最高的 k 个标记
    top_p=0.9, #表示在采样时的重复惩罚（repetition penalty）。较高的值会鼓励模型生成更加多样化的文本，较低的值会鼓励模型生成更加一致的文本
    do_sample=True, #如果设置为 True，则生成时将使用采样策略，从模型的概率分布中随机选择下一个标记。如果设置为 False，则将使用贪婪（greedy）策略选择概率最高的标记。
    num_beams=1, # 较大的 num_beams 值会增加生成的多样性，因为更多的候选项被保留并参与下一步的扩展。然而，较大的值也会增加计算开销
    repetition_penalty=1.1, #同top_p
    max_new_tokens=400 # 参数用于限制生成的文本中新增标记的数量。标记可以是单词、子词或字符，具体取决于所使用的模型和分词器，参数是相对于输入上下文而言的。如果输入上下文已经包含一些标记，那么生成的文本中新增标记的数量将计算为生成文本和输入上下文之间的差异
)
    
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

def resize_model_vocab_size(base_model,tokenizer):
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

def load_model(model_path,tokenizer_path=None,llama=False,lora_model=None,
               load_in_4bit=False,
               load_in_8bit=False,
               use_vllm=False,
               gpus=""):
    if tokenizer_path == None:
        tokenizer_path = model_path

    load_type = torch.float16
    if llama:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=True)
        print("llama special token:bos",tokenizer.bos_token,tokenizer.encode(tokenizer.bos_token))
        print("llama special token:eos",tokenizer.eos_token,tokenizer.encode(tokenizer.bos_token))
        base_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map='auto',
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=load_type
                )
            )      
        # print("llama special token:bos",base_model.config.bos_token_id)
        # print("llama special token:eos",base_model.config.eos_token_id)  
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            load_type = torch.bfloat16

        base_model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                            device_map="auto", 
                                                            trust_remote_code=True,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=load_type)
        global generation_config
        generation_config = GenerationConfig.from_pretrained(model_path, 
                                                             trust_remote_code=True)

    if lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, lora_model,
                                          torch_dtype=load_type,device_map='auto',).half()
    else:
        model = base_model

    print("generation_config {}",generation_config)
    return model.to(device),tokenizer



def qwen_chat(raw_text,model,tokenizer):
    global history
    response, history = model.chat(tokenizer, raw_text, history=history,generation_config = generation_config)
    return response

def do_generate(input_text,model,tokenizer,use_vllm=False):
    
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids

    input_ids = input_ids.to(device)
    generation_output = model.generate(
        input_ids = input_ids,
        generation_config = generation_config
    )

    response_ids = generation_output[0]
    output = tokenizer.decode(response_ids.cpu(),skip_special_tokens=True)
    response = output
    return response

def chat(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.你是一个乐于助人的助手。",
        append_history: bool = True,
        chat_format: str = "chatml",
        stop_words_ids: Optional[List[List[int]]] = None,
        max_window_size: int = 6144,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Tuple[str, HistoryType]:
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=chat_format
        )
        stop_words_ids.extend(get_stop_words_ids(
            chat_format, tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(device)
        outputs = model.generate(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    stop_words_ids=stop_words_ids
                )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=chat_format,
            verbose=False,
            errors='replace'
        )

        if append_history:
            history.append((query, response))

        return response, history
    