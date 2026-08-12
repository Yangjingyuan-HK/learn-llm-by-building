"""
数据加载

PretrainDataset: 读 {"text": "..."} → [BOS] tokens [EOS] padding, 全程算 loss
SFTDataset:     读 {"conversations": [...]} → chat_template 展开, 只算 assistant 回复的 loss
"""
import json
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = json.loads(line).get('text', '').strip()
                if text:
                    self.data.append(text)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = [self.tokenizer.bos_token_id] + \
              self.tokenizer.encode(self.data[idx], add_special_tokens=False)[:self.max_length - 2] + \
              [self.tokenizer.eos_token_id]
        labels = list(ids)
        ids += [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        labels += [-100] * (self.max_length - len(labels))
        return torch.tensor(ids), torch.tensor(labels)


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                convs = json.loads(line).get('conversations', [])
                if convs:
                    self.data.append(convs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = []
        for c in self.data[idx]:
            role = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}.get(
                c.get('from', c.get('role', '')), 'user')
            messages.append({"role": role, "content": c.get('value', c.get('content', ''))})

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)[:self.max_length]

        labels = [-100] * len(full_ids)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'assistant':
                prefix = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True)
                start = len(self.tokenizer.encode(prefix, add_special_tokens=False))
                for j in range(start, len(full_ids)):
                    labels[j] = full_ids[j]

        # padding
        ids = full_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(full_ids))
        labels += [-100] * (self.max_length - len(labels))
        return torch.tensor(ids), torch.tensor(labels)


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])