import torch
import json
from transformers import BertTokenizer

class REDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                try:
                    sample = json.loads(line)
                    self.data.append(sample)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")

        self.tokenizer = tokenizer
        # 假设从 duie_schema 中获取所有关系类型
        schema_path = 'data/duie_schema.json'  # 根据实际路径修改
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        relations = [item['predicate'] for item in schema]
        self.label2id = {relation: idx for idx, relation in enumerate(relations)}
        self.label2id['no_relation'] = len(relations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        spo_list = sample.get('spo_list', [])
        if spo_list:
            # 取第一个三元组作为示例，可根据需求修改
            spo = spo_list[0]
            subject = spo['subject']
            object_ = spo['object']['@value']
            predicate = spo['predicate']
            label_id = self.label2id.get(predicate, self.label2id['no_relation'])

            subject_start = text.find(subject)
            object_start = text.find(object_)

            if subject_start != -1 and object_start != -1:
                e1_start = subject_start
                e2_start = object_start
            else:
                e1_start = 0
                e2_start = 1
        else:
            label_id = self.label2id['no_relation']
            e1_start = 0
            e2_start = 1

        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        input_ids = encoding['input_ids'].squeeze()
        e1_pos = torch.tensor([e1_start])
        e2_pos = torch.tensor([e2_start])

        return {
            'input_ids': input_ids,
            'e1_pos': e1_pos,
            'e2_pos': e2_pos,
            'label': label_id
        }