from dataclasses import dataclass, field
import datasets
import transformers
from typing import Optional, List, Dict, Any, Mapping
from transformers.utils.versions import require_version
from transformers.testing_utils import CaptureLogger
from itertools import chain
from pathlib import Path
import os
from datasets import load_dataset, concatenate_datasets
import sys
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from llm_model import create_tokenizer,ModelArguments
import logging
logger = logging.getLogger(__name__)
block_size = 512

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

# 使用分词器对输入的文本进行分词
count = 0
def tokenize_function(tokenizer):
    def do_tokenize(examples):
        # examples = {'text': []}
        # 默认加载1000行,text对应的数组按照行加载
        global count
        if count == 0:
            print("tokenize_function:examples----",len(examples["text"][101]))
        with CaptureLogger(tok_logger) as cl:
            # output = { "input_ids": [],attention_mask:[]}
            output = tokenizer(examples["text"])
            if count == 0:
                print("tokenize_function---tokenizer",len(output["input_ids"][101]),len(output["attention_mask"][101]))
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        
        count += 1
        # print("tokenize_function:count",count)
        return output
    return do_tokenize
gcount = 0
# 将token之后的数据,按blocksize组织
def group_texts(examples):
    global gcount
    # Concatenate all texts.
    if gcount == 0:
        print("group_texts:keys:",len(examples["input_ids"][101]),len(examples["attention_mask"][101]))
        print("group_texts:keys:",len(examples.keys()),type(examples.keys()))
        for k in examples.keys():
            print("iner examples.keys",k)
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    # result = {"input_ids":[512*99],"attention_mask":[512*99]}
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    if gcount == 0:
        print("concatenated_examples:keys:",len(concatenated_examples["input_ids"]),len(concatenated_examples["attention_mask"]),type(concatenated_examples))
        print("total_length:keys:",total_length)
        print("result",len(result["input_ids"]),len(result["input_ids"][0]),)
    gcount += 1
    return result

def determine_block_size(data_args, tokenizer):
    """
    Determines the block size based on the provided data arguments and tokenizer's maximum length.

    :param data_args: The DataTrainingArguments object containing the user-specified data arguments.
    :param tokenizer: The Hugging Face tokenizer used to tokenize the input text.
    :return: The determined block size.
    """
    # Determine the block size
    global block_size

    block_size = data_args.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    print("block_size:-----",block_size)
    return block_size

# 返回
# Dataset({
#     features: ['input_ids', 'attention_mask', 'labels'],
#     num_rows: 7532
# })
def preprocess_dataset(data_args,block_size,tokenize):
    
    lm_datasets = []
    path = Path(data_args.dataset_dir)
    files = [file.name for file in path.glob("*.txt")]
    print("files---",files)
    for idx, file in enumerate(files):
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(data_args.data_cache_dir, filename+f"_{block_size}")
        print("cache_path---",cache_path)
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
            print("processed_dataset---",processed_dataset)
            logger.info(f'training datasets-{filename} has been loaded from disk')
        except Exception:
            cache_dir = os.path.join(data_args.data_cache_dir, filename+f"_text_{block_size}")
            print("cache_dir---",cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
            logger.info(f"{file} has been loaded")
            do_tokenize = tokenize_function(tokenize)
            # 分词处理
            tokenized_dataset = raw_dataset.map(
                do_tokenize,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns="text",
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names = {k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                desc="Running tokenizer on dataset",
            )
            print("tokenized_dataset---",tokenized_dataset)
            grouped_datasets = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                desc=f"Grouping texts in chunks of {block_size}",
            )
            print("grouped_datasets---",grouped_datasets)
            processed_dataset = grouped_datasets
            processed_dataset.save_to_disk(cache_path)
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            print("features.type---",lm_datasets.features.type)
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
    lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)
    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets["test"]
    print("train_dataset---",train_dataset)
    print("eval_dataset---",eval_dataset)
    return train_dataset,eval_dataset

if __name__ == "__main__":
    parser = HfArgumentParser((DataTrainingArguments,ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args,token_arg = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args,token_arg = parser.parse_args_into_dataclasses()

    tokenizer = create_tokenizer(token_arg)
    determine_block_size(data_args,tokenizer)
    preprocess_dataset(data_args,block_size,tokenizer)
