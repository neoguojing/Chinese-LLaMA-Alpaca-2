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

@dataclass
class NLPExample:
    text: str
    label: str

@dataclass
class NLPTrainData:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[int]

def generate_tokenize_func(tokenizer: PreTrainedTokenizer,
                           data_format:str="text"):
    if data_format == "text":
        return lambda examples: tokenizer(examples["text"])
    elif data_format == "json":
        return lambda examples: tokenizer(examples["text"])
    elif data_format == "qwen":
        return lambda examples: tokenizer(examples["text"])
    
block_size = 512
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

class NLPDataBuilder:
    def __init__(self, dataset_dir: str, tokenizer: PreTrainedTokenizer,
                 block_size: int = 512,
                 cache_dir: str = None,data_format: str = "json",
                 json_field: str = "data",num_of_procs:int = 8,
                 validation_split_percentage: float=0.05):
        self.dataset_dir = dataset_dir
        self.block_size = block_size
        self.cache_dir = cache_dir
        self.tokenizer =tokenizer
        self.data_format = data_format
        self.json_field = json_field

    def build_dataset(self, train_ratio: float, test_ratio: float, 
                      val_ratio: float, batch_size: int) -> DataLoader:
        path = Path(self.dataset_dir)
        suffix = "json"
        if self.data_format == "text":
            suffix = "txt"
        pattern = "*."+suffix

        files = [file.name for file in path.glob(pattern)]
        for idx, file in enumerate(files):
            if self.cache_dir != None:
                tokenized_dataset = self._load_tokenized_data_from_cache(file)
            if tokenized_dataset == None:
                raw_dataset = self._load_raw_data(file)
                tokenized_dataset = self._tokenize_data(raw_dataset)
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
        raw_dataset = load_dataset(self.data_format, data_files=data_file, 
                                   cache_dir=cache_dir, keep_in_memory=False,
                                   field=self.json_field)
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
        lm_datasets = tokenized_dataset.train_test_split(test_size = self.validation_split_percentage)
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

# Example usage
file_paths = ['data/train.txt', 'data/test.txt', 'data/val.txt']
tokenizer_name = 'bert-base-uncased'
max_length = 128
batch_size = 32

data_builder = NLPDataBuilder(file_paths, tokenizer_name, max_length)
train_dataloader, test_dataloader, val_dataloader = data_builder.build_dataset(0.8, 0.1, 0.1, batch_size)

for batch in train_dataloader:
    input_ids, attention_mask, labels = batch
    # Process and train the batch