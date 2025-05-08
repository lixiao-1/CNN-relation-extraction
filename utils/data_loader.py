import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class REDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id):
        self.tokenizer = tokenizer
        self.data = []
        self.label2id = label2id
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    text = item['text']
                    spo_list = item.get('spo_list', [])
                    for spo in spo_list:
                        predicate = spo['predicate']
                        if predicate not in self.label2id:
                            print(f"Warning: Label {predicate} not found in label2id mapping for item {idx} of {file_path}. Skipping...")
                            continue
                        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
                        input_ids = encoding['input_ids'].squeeze(0)
                        attention_mask = encoding['attention_mask'].squeeze(0)
                        label_id = torch.tensor(self.label2id[predicate])
                        self.data.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'label': label_id
                        })
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]