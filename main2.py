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
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)
        tokens = batch
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            logits = model(batch)
            loss = criterion(logits, tokens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        writer.log({"loss": loss.detach().cpu().item(), 'lr': lr,})



def trainer(num_epochs, model, optimizer, scheduler,
             loss, scaler, writer, dataloader, model_cfg={}, device='cpu'):
    train_losses = []
    model = model.to(device)
    for i in range(num_epochs):
        train_epoch(model, optimizer, scheduler, loss,
                                  scaler, writer, dataloader, device)
        checkpoint = {
            'config': model_cfg,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_{i}.pth')
        
    return train_losses

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch.backends.cudnn.deterministic = False
torch.cuda.empty_cache()


with open('cfgs/cfg.json') as f:
    cfg = json.load(f)

model_cfg = ModelConfig(**cfg['ModelConfig'])

dataset = CustomDataset(**cfg['DatasetConfig'])

dataloader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn, num_workers=8, 
                        shuffle=True, drop_last=True)
print(len(dataloader))
device = torch.device('cuda:0')

scaler = torch.cuda.amp.GradScaler(enabled=True)

checkpoint = torch.load('checkpoints/checkpoint_0.pth')

model = GPT2(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['state_dict'])
aaa = count_parameters(model)

aaa = count_parameters(model)
print(aaa)
epochs=4

criterion = LossLM(model_cfg, label_smoothing=0)

model.eval()

sum_loss = 0.
ks = 0
i = 0

for data in tqdm(dataloader):
    data = data.to(device)
    logits = model(data)
    loss = criterion(logits, data)
    sum_loss +=  20* loss.cpu().item()
    ks += 20
    i += 1
    if i > 6000:
        break
print(sum_loss/ks)
