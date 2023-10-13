import argparse
import json, os

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

history_max_len = 1000  # 模型记忆的最大token长度
history_token_ids = torch.tensor([[]], dtype=torch.long)

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--alpha', type=str, default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--load_in_4bit', action='store_true', help="Load the LLM in the 4bit mode")
parser.add_argument("--use_vllm", action='store_true', help="Use vLLM as back-end LLM service.")
parser.add_argument('--system_prompt', type=str, default=DEFAULT_SYSTEM_PROMPT, help="The system prompt of the prompt template.")
parser.add_argument('--negative_prompt', type=str, default=None, help="Negative prompt in CFG sampling.")
parser.add_argument('--guidance_scale', type=float, default=1.0, help="The guidance scale for CFG sampling. CFG is enabled by setting `guidance_scale > 1`.")
parser.add_argument('--llama',action='store_true', help="is llama like model")
args = parser.parse_args()

if args.guidance_scale > 1:
    try:
        from transformers.generation import UnbatchedClassifierFreeGuidanceLogitsProcessor
    except ImportError:
        raise ImportError("Please install the latest transformers (commit equal or later than d533465) to enable CFG sampling.")

if args.use_vllm:
    if args.lora_model is not None:
        raise ValueError("vLLM currently does not support LoRA, please merge the LoRA weights to the base model.")
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("vLLM currently does not support quantization, please use fp16 (default) or unuse --use_vllm.")
    if args.only_cpu:
        raise ValueError("vLLM requires GPUs with compute capability not less than 7.0. If you want to run only on CPU, please unuse --use_vllm.")
    if args.guidance_scale > 1:
        raise ValueError("guidance_scale > 1, but vLLM does not support CFG sampling. Please unset guidance_scale. ")
    
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
    
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoConfig,AutoModelForCausalLM,AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import  PeftModel
if args.use_vllm:
    from vllm import LLM, SamplingParams

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
if not args.only_cpu:
    apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

if args.use_vllm:
    generation_config = dict(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        max_tokens=400,
        presence_penalty=1.0,
    )
else:
    
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

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

def resize_model_vocab_size(base_model,tokenizer):
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

def load_model(args):
    load_type = torch.float16
    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
        base_model = LlamaForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map='auto',
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                    bnb_4bit_compute_dtype=load_type
                )
            )        
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, 
                                                            device_map="auto", 
                                                            trust_remote_code=True,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=load_type)
        global generation_config
        generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)

    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',).half()
    elif args.use_vllm:
        print("loading vllm model")
        model = LLM(model=args.base_model,
            tokenizer=args.tokenizer_path,
            tokenizer_mode='slow',
            tensor_parallel_size=len(args.gpus.split(',')))
    else:
        model = base_model

    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    return model.to(device),tokenizer

# 包装输入，用于支持多轮对话
def wrap_history(token_ids):
    global history_token_ids
    history_token_ids = torch.concat((history_token_ids, token_ids.cpu()), dim=1)
    model_input_ids = history_token_ids[:, -history_max_len:]
    return model_input_ids

def do_generate(input_text,negative_text,model,tokenizer):
    if args.use_vllm:
        output = model.generate([input_text], SamplingParams(**generation_config), use_tqdm=False)
        response = output[0].outputs[0].text
    else:
        input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
            input_ids = torch.concat([input_ids, eos_token_id], dim=1)

        input_ids = wrap_history(input_ids).to(device)
        if args.guidance_scale ==1:
            generation_output = model.generate(
                input_ids = input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config
            )
        else: # enable CFG sampling
            # CFG Scale ：
            # 参数越大，生成的图像与文本提示的相关性越高，但可能会失真。
            # 数值越小，相关性则越低，越有可能偏离提示或输入图像，但质量越好。
            if negative_text is None:
                negative_prompt_ids = None
                negative_prompt_attention_mask = None
            else:
                negative_inputs = tokenizer(negative_text,return_tensors="pt")
                negative_prompt_ids = negative_inputs["input_ids"].to(device)
                negative_prompt_attention_mask = negative_inputs["attention_mask"].to(device)
            generation_output = model.generate(
                input_ids = input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config,
                guidance_scale = args.guidance_scale,
                negative_prompt_ids = negative_prompt_ids,
                negative_prompt_attention_mask = negative_prompt_attention_mask
            )

        # s = generation_output[0]
        # output = tokenizer.decode(s,skip_special_tokens=True)

        model_input_ids_len = input_ids.size(1)
        response_ids = generation_output[:, model_input_ids_len:]
        wrap_history(response_ids)
        output = tokenizer.batch_decode(response_ids)
        print(output)
        print("\n")
        # 处理提示词
        if args.with_prompt:
            response = output.split("[/INST]")[-1].strip()
        else:
            response = output

    return response

if __name__ == '__main__':

    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model

    model,tokenizer = load_model(args)
    if device==torch.device('cpu'):
        model.float()
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference")
            while True:
                # 用户输入
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                # 处理提示词
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text, system_prompt=args.system_prompt)
                    negative_text = None if args.negative_prompt is None \
                        else generate_prompt(instruction=raw_input_text, system_prompt=args.negative_prompt)
                else:
                    input_text = raw_input_text
                    negative_text = args.negative_prompt

                response = do_generate(input_text,negative_text,model,tokenizer)
                print("Response: ",response)
                print("\n")

