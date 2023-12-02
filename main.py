import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece
from model import ModelConfig, FinalModel
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

def loss_lm(logits, tokens):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def train_epoch(model, optimizer, dataloader, device):
    f_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        tokens = batch
        logits = model(batch)
        loss = loss_lm(logits, tokens)
        loss.backward()
        optimizer.step()
        f_loss += loss.detach().cpu().item()
    return f_loss


def trainer(num_epochs, model, optimiser, dataloader, device='cpu'):
    train_losses = []
    model = model.to(device)
    for i in range(num_epochs):
        train_loss = train_epoch(model, optimiser, dataloader, device)
        train_losses.append(train_loss)
        print(i, train_loss)
    return train_losses






encoder = sentencepiece.SentencePieceProcessor(model_file='m.model')

dataset = CustomDataset(encoder, 'dataset.json')

dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)

cfg = ModelConfig()

model = FinalModel(cfg)

device = torch.device('cuda:0')

optimiser = torch.optim.Adam(model.parameters(), lr=3e-5)

losses = trainer(5000, model, optimiser, dataloader, device)

print(losses[:5])
print(losses[-5:])