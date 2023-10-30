import os
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List
from dataclasses import dataclass, field
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    Dataset, 
    DatasetDict
)
from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Any, Mapping,Union
import torch
from transformers.trainer_pt_utils import LabelSmoother
import pdb
@dataclass
class NLPExample:
    text: str
    label: str

@dataclass
class NLPTrainData:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[int]

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_INDEX = -100
PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    )
def generate_tokenize_func(tokenizer: PreTrainedTokenizer,
                           max_seq_length: int,
                           data_format:str="text",):
    '''
    qwen:
    DatasetDict({
        train: Dataset({
            features: ['conversations', 'id'],
            num_rows: 822
        })
    })

    '''
    if data_format == "text":
        return lambda examples: tokenizer(examples["text"])
    
    elif data_format == "llama":
        def tokenization(examples):
            sources = []
            targets = []
            prompt = PROMPT_TEMPLATE
            for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
                if input is not None and input !="":
                    instruction = instruction+'\n'+input
                source = prompt.format_map({'instruction':instruction})
                target = f"{output}{tokenizer.eos_token}"

                sources.append(source)
                targets.append(target)

            tokenized_sources = tokenizer(sources,return_attention_mask=False)
            tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

            all_input_ids = []
            all_labels = []
            for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
                assert len(input_ids) == len(labels)
                all_input_ids.append(input_ids)
                all_labels.append(labels)

            results = {
                'input_ids':all_input_ids,
                'labels': all_labels,
                'attention_mask': all_input_ids.ne(tokenizer.pad_token_id)
            }
            return results
        return tokenization
    elif data_format == "qwen":
        system_message = "You are a helpful assistant."
        def tokenization(examples):
            pdb.set_trace()
            roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
            im_start = tokenizer.im_start_id
            im_end = tokenizer.im_end_id
            nl_tokens = tokenizer('\n').input_ids
            _system = tokenizer('system').input_ids + nl_tokens
            _user = tokenizer('user').input_ids + nl_tokens
            _assistant = tokenizer('assistant').input_ids + nl_tokens

            # Apply prompt templates
            sources = examples["conversations"]
            input_ids, targets = [], []
            for i, source in enumerate(sources):
                if roles[source[0]["from"]] != roles["user"]:
                    source = source[1:]

                input_id, target = [], []
                system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
                input_id += system
                target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
                assert len(input_id) == len(target)
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                    input_id += _input_id
                    if role == '<|im_start|>user':
                        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                    elif role == '<|im_start|>assistant':
                        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                            _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                    else:
                        raise NotImplementedError
                    target += _target
                assert len(input_id) == len(target)
                input_id += [tokenizer.pad_token_id] * (max_seq_length - len(input_id))
                target += [IGNORE_TOKEN_ID] * (max_seq_length - len(target))
                input_ids.append(input_id[:max_seq_length])
                targets.append(target[:max_seq_length])
            input_ids = torch.tensor(input_ids, dtype=torch.int)
            targets = torch.tensor(targets, dtype=torch.int)

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )
        return tokenization

def generate_group_func(block_size: int= 512):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    return group_texts

class NLPDataBuilder:
    def __init__(self, dataset_dir: str, tokenizer: PreTrainedTokenizer,
                 block_size: int = 512,
                 cache_dir: str = None,data_format: str = "qwen",
                 num_of_procs:int = 8,
                 validation_split_percentage: float=0.05):
        self.dataset_dir = dataset_dir
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.tokenizer =tokenizer
        self.data_format = data_format
        self.num_of_procs = num_of_procs
        self.validation_split_percentage = validation_split_percentage

    def build_dataset(self):
        path = Path(self.dataset_dir)
        suffix = "json"
        if self.data_format == "text":
            suffix = "txt"
        elif self.data_format == "qwen":
            self.data_format = "json"
            suffix = "qwen"
        pattern = "*."+suffix

        files = [file.name for file in path.glob(pattern)]
        print(files)
        for idx, file in enumerate(files):
            pdb.set_trace()
            if self.cache_dir != None:
                tokenized_dataset = self._load_tokenized_data_from_cache(file)
            if tokenized_dataset == None:
                raw_dataset = self._load_raw_data(file)
                tokenized_dataset = self._tokenize_data(raw_dataset)
                if self.data_format == "text":
                    tokenized_dataset = self._split_data(tokenized_dataset)
                self._save_tokenized_data_to_cache(file,tokenized_dataset)
            train_dataset,eval_dataset = self._group_data(tokenized_dataset)

        return train_dataset, eval_dataset
    
    def _load_raw_data(self,file:str) -> Union[Dataset, DatasetDict]:
        path = Path(self.dataset_dir)
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_dir = os.path.join(self.cache_dir, filename+f"_text_{self.block_size}")
        os.makedirs(cache_dir, exist_ok=True)
        print("_load_raw_data",data_file)
        raw_dataset = load_dataset(self.data_format, data_files=data_file, 
                                   cache_dir=cache_dir, keep_in_memory=False)
        return raw_dataset

    def _load_tokenized_data_from_cache(self,file: str) -> Union[Dataset, DatasetDict]:
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(self.cache_dir, filename+f"_{self.block_size}")
        os.makedirs(cache_path, exist_ok=True)
        try:
            tokenized_dataset = load_from_disk(cache_path, keep_in_memory=False)
        except Exception:
            return None
        return tokenized_dataset
    
    def _save_tokenized_data_to_cache(self, file: str,
                                      tokenized_dataset: Union[Dataset, DatasetDict]):
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(self.cache_dir, filename+f"_{self.block_size}")
        os.makedirs(cache_path, exist_ok=True)
        tokenized_dataset.save_to_disk(cache_path)
      

    def _preprocess_data(self, data: List[NLPExample]) -> List[NLPExample]:
        processed_data = []
        for example in data:
            # Preprocess data (e.g., data cleaning, normalization, etc.)
            # Append preprocessed example to processed_data
            pass
        return processed_data

    def _tokenize_data(self, raw_dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        do_tokenize = generate_tokenize_func(self.tokenizer,self.data_format)
        tokenized_dataset = raw_dataset.map(
                do_tokenize,
                batched=True,
                num_proc=self.num_of_procs,
                remove_columns="",
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names = {k: os.path.join(self.cache_dir, 'tokenized.arrow') for k in raw_dataset},
                desc="Running tokenizer on dataset",
            )
        return tokenized_dataset

    def _split_data(self, tokenized_dataset: Union[Dataset, DatasetDict],
                    ) -> Union[Dataset, DatasetDict]:
        group_texts = generate_group_func(self.block_size)
        grouped_datasets = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=self.num_of_procs,
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names = {k: os.path.join(self.cache_dir, 'grouped.arrow') for k in tokenized_dataset},
            desc=f"Grouping texts in chunks of {self.block_size}",
        )
        return grouped_datasets
    
    def _group_data(self, tokenized_dataset: Union[Dataset, DatasetDict],
                    ):
        lm_datasets = tokenized_dataset['train']
        lm_datasets = lm_datasets.train_test_split(test_size = self.validation_split_percentage)
        train_dataset = lm_datasets['train']
        eval_dataset = lm_datasets["test"]
        print("train_dataset---",train_dataset)
        print("eval_dataset---",eval_dataset)
        return train_dataset,eval_dataset

    def _create_dataloader(self, data: NLPTrainData, batch_size: int) -> DataLoader:
        dataset = NLPDataset(data.input_ids, data.attention_mask, data.labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

@dataclass
class NLPDataset(Dataset):
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_id = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        label = self.labels[index]
        return input_id, attention_mask, label

if __name__ == "__main__":
    from llm_model import create_tokenizer
    tokenizer = create_tokenizer("../../model/chinese/Qwen-7B-Chat/",8192)
    builder = NLPDataBuilder("../../dataset/chat/",tokenizer,cache_dir=".")
    builder.build_dataset()
    