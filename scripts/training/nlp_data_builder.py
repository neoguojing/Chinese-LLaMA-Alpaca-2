from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List
from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets

@dataclass
class NLPExample:
    text: str
    label: str

@dataclass
class NLPTrainData:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[int]

class NLPDataBuilder:
    def __init__(self, file_paths: List[str], tokenizer_name: str, max_length: int):
        self.file_paths = file_paths
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def build_dataset(self, train_ratio: float, test_ratio: float, val_ratio: float, batch_size: int) -> DataLoader:
        dataset = self._load_data()
        dataset = self._preprocess_data(dataset)
        dataset = self._tokenize_data(dataset)
        train_data, test_data, val_data = self._split_data(dataset, train_ratio, test_ratio, val_ratio)
        train_dataloader = self._create_dataloader(train_data, batch_size)
        test_dataloader = self._create_dataloader(test_data, batch_size)
        val_dataloader = self._create_dataloader(val_data, batch_size)
        return train_dataloader, test_dataloader, val_dataloader

    def _load_data(self) -> List[NLPExample]:
        data = []
        for file_path in self.file_paths:
            # Load data from file
            # Create NLPExample objects and append to data
            pass
        return data

    def _preprocess_data(self, data: List[NLPExample]) -> List[NLPExample]:
        processed_data = []
        for example in data:
            # Preprocess data (e.g., data cleaning, normalization, etc.)
            # Append preprocessed example to processed_data
            pass
        return processed_data

    def _tokenize_data(self, data: List[NLPExample]) -> NLPTrainData:
        input_ids = []
        attention_mask = []
        labels = []
        for example in data:
            # Tokenize text using tokenizer
            # Convert tokens to input_ids and attention_mask
            # Append input_ids, attention_mask, and label to respective lists
            pass
        return NLPTrainData(input_ids, attention_mask, labels)

    def _split_data(self, data: NLPTrainData, train_ratio: float, test_ratio: float, val_ratio: float) -> tuple:
        # Split data into train, test, and validation sets based on the given ratios
        # Return train_data, test_data, and val_data
        pass

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