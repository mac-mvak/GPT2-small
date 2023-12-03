from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece
from src.model import ModelConfig, GPT2
import json
import pickle
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, max_tok, dataset_path, starts_path, **kwargs):
        super().__init__()
        self.texts = torch.load(dataset_path)
        self.starts = torch.load(starts_path)
        self.max_tok = max_tok
        self.bos = torch.LongTensor([1])
        self.eos = torch.LongTensor([2])

    def __len__(self):
        return self.starts.shape[0]
    
    def __getitem__(self, i):
        start = self.starts[i]
        text = self.texts[start[0]:start[1]].long()
        text = text[:self.max_tok]
        text = torch.cat([self.bos, text, self.eos])
        return {'tokens': text.unsqueeze(0)}


def adder(vec, v):
    if vec is None:
        vec = v
    else:
        size_1, size_2 = vec.shape[-1], v.shape[-1]
        pad = size_1 - size_2
        vec = nn.functional.pad(vec, (0, max(-pad, 0)), value=3)
        v = nn.functional.pad(v, (0, max(pad, 0)), value=3)
        vec = torch.cat([vec, v])
    return vec

def collate_fn(data):
    tokens = None
    for vec in data:
        tokens = adder(tokens, vec['tokens'])
    return tokens

