import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_size: int = 1000
    len_vocab: int = 1000
    padding_idx: int = 3
    max_len: int = 1024
    init_s: float = 0.02
    num_heads: int = 8
    fc_size: int = 1024
    n_blocks: int = 5

def init_module(module, cfg: ModelConfig):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, std=cfg.init_s)
        elif 'bias' in name:
            nn.init.zeros_(param)



class PositionalEmbedding(nn.Module):
    def __init__(self, cfg:ModelConfig):
        super().__init__()
        self.pos_weight = nn.Parameter(torch.empty((cfg.max_len, cfg.embed_size)))

    def forward(self, tokens):
        pos_embeds = self.pos_weight[:tokens.shape[1], :]
        return pos_embeds.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_size)
        self.attention = nn.MultiheadAttention(cfg.embed_size, cfg.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.embed_size)
        self.fc1 = nn.Linear(cfg.embed_size, cfg.fc_size)
        self.fc2 = nn.Linear(cfg.fc_size, cfg.embed_size)
        

    def forward(self, X_in):
        normal_X_in = self.ln1(X_in)
        mask = nn.Transformer.generate_square_subsequent_mask(sz=X_in.shape[1]).to(X_in.device)
        attn, _ = self.attention(normal_X_in, normal_X_in, normal_X_in, need_weights=False,
                              attn_mask=mask, is_causal=True)
        X_mid = attn + X_in
        normal_X_mid = self.ln2(X_mid)
        fc_X_mid = self.fc2(
            nn.functional.gelu(self.fc1(normal_X_mid))
        )
        X_out = fc_X_mid + X_mid
        return X_out
        

class FinalModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.len_vocab, cfg.embed_size) #, padding_idx=cfg.padding_idx)
        self.pos_embed = PositionalEmbedding(cfg)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_blocks)] )
        self.ln_final = nn.LayerNorm(cfg.embed_size)
        self.fc_final = nn.Linear(cfg.embed_size, cfg.len_vocab)
        init_module(self.pos_embed, cfg)
        init_module(self.blocks, cfg)
        init_module(self.fc_final, cfg)
    
    def forward(self, tokens):
        embed = self.emb(tokens)
        pos_embed = self.pos_embed(tokens)
        f_embed = embed + pos_embed
        blocks_out = self.blocks(f_embed)
        norm_blocks_out = self.ln_final(blocks_out)
        logits = self.fc_final(norm_blocks_out)
        return logits




