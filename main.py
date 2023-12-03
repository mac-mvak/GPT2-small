from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece
from model import ModelConfig, GPT2
import json
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, encoder, json_path):
        super().__init__()
        self.encoder = encoder
        with open(json_path) as f:
            self.texts = json.load(f)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        text = self.texts[i]['story']
        tokens = [self.encoder.bos_id()]
        tokens += self.encoder.encode(text) + [self.encoder.eos_id()]
        return {'tokens':  torch.LongTensor(tokens).unsqueeze(0)}

class LossLM(nn.CrossEntropyLoss):
    def __init__(self, cfg: ModelConfig,
                  weight: Tensor | None = None, size_average=None, ignore_index: int = -100,
                  reduce=None, reduction: str = 'mean', label_smoothing: float = 0.1) -> None:
        ignore_index = cfg.padding_idx
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, logits, tokens):
        loss = super().forward(
            logits.view(-1, logits.shape[-1]), tokens.view(-1)
        )
        return loss


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


def train_epoch(model, optimizer, loss, dataloader, device):
    f_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        tokens = batch
        logits = model(batch)
        loss = loss(logits, tokens)
        loss.backward()
        optimizer.step()
        f_loss += loss.detach().cpu().item()
    return f_loss


def trainer(num_epochs, model, optimiser, loss, dataloader, device='cpu'):
    train_losses = []
    model = model.to(device)
    for i in range(num_epochs):
        train_loss = train_epoch(model, optimiser, loss, dataloader, device)
        train_losses.append(train_loss)
        print(i, train_loss)
    return train_losses






encoder = sentencepiece.SentencePieceProcessor(model_file='m.model')

dataset = CustomDataset(encoder, 'dataset.json')

dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)

cfg = ModelConfig()

model = GPT2(cfg)

device = torch.device('cuda:0')

loss = LossLM(cfg)

optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

losses = trainer(5000, model, optimiser, loss, dataloader, device)

print(losses[:5])
print(losses[-5:])