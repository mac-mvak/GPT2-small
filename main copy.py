from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import CustomDataset, collate_fn
import sentencepiece
from src.model import GPT2
from src.config import ModelConfig
import json
from src.logger import Writer
from tqdm import tqdm


class LossLM(nn.CrossEntropyLoss):
    def __init__(self, cfg: ModelConfig,
                  weight: Tensor | None = None, size_average=None, ignore_index: int = -100,
                  reduce=None, reduction: str = 'mean', label_smoothing: float = 0.1) -> None:
        ignore_index = cfg.padding_idx
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, logits, tokens):
        a = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        b = tokens[:, 1:].reshape(-1)
        loss = super().forward(a, b)
        return loss



def train_epoch(model, optimizer, scheduler, criterion, scaler, writer, dataloader, device):
    f_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        tokens = batch
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            logits = model(batch)
            loss = criterion(logits, tokens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        u = scheduler.get_last_lr()[0]
        writer.log({"loss": loss.detach().cpu().item(), 'lr': u})



def trainer(num_epochs, model, optimizer, scheduler,
             loss, scaler, writer, dataloader, model_cfg={}, device='cpu'):
    train_losses = []
    model = model.to(device)
    for i in tqdm(range(num_epochs)):
        train_epoch(model, optimizer, scheduler, loss,
                                  scaler, writer, dataloader, device)
        checkpoint = {
            'config': model_cfg,
            'state_dict': model.state_dict()
        }
        #torch.save(checkpoint, f'checkpoints/checkpoint_{i}.pth')
        
    return train_losses

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




with open('cfgs/cfg.json') as f:
    cfg = json.load(f)

model_cfg = ModelConfig(**cfg['ModelConfig'])

dataset = CustomDataset(**cfg['DatasetConfig'])

dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, num_workers=5, 
                        shuffle=True, drop_last=True)
print(len(dataloader))
scaler = torch.cuda.amp.GradScaler(enabled=True)

checkpoint = torch.load('checkpoints/checkpoint.pth')

model = GPT2(checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
aaa = count_parameters(model)

u = dataset[2]['tokens']

model.eval()
ans_u = model(u)
ans_u = ans_u.squeeze()
u = u.squeeze()
armax = torch.argmax(ans_u, dim=-1)
print(u[1:]==armax[:-1])

