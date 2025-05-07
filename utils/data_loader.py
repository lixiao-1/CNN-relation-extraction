import torch
import json
from transformers import BertTokenizer

class REDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file: {e}")
                self.data = []
        self.tokenizer = tokenizer
        self.label2id = {
            'address': 0,
            'book': 1,
            'company': 2,
            'game': 3,
            'government': 4,
            'movie': 5,
            'name': 6,
            'organization': 7,
            'position': 8,
            'scene': 9,
            'no_relation': 10
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        label = sample['label']
        label_id = self.label2id[label]
        entities = sample['entities']

        # 假设 entities 是一个包含实体信息的列表，每个实体有 'start' 和 'end' 字段
        if len(entities) >= 2:
            e1_start = entities[0]['start']
            e2_start = entities[1]['start']
            encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = encoding['input_ids'].squeeze()
            # 找到实体在 input_ids 中的位置
            e1_pos = torch.tensor([e1_start])
            e2_pos = torch.tensor([e2_start])
        else:
            # 如果实体数量不足，设置默认值
            encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = encoding['input_ids'].squeeze()
            e1_pos = torch.tensor([0])
            e2_pos = torch.tensor([1])

        return {
            'input_ids': input_ids,
            'e1_pos': e1_pos,
            'e2_pos': e2_pos,
            'label': label_id
        }