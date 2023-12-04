import torch
import torch.nn as nn
from .config import ModelConfig


def init_module(module, cfg: ModelConfig):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, std=cfg.init_s)
        elif 'bias' in name:
            nn.init.zeros_(param)



class PositionalEmbedding(nn.Module):
    def __init__(self, cfg:ModelConfig):
        super().__init__()
        self.pos_weight = nn.Parameter(torch.randn((cfg.max_len, cfg.embed_size)))

    def forward(self, tokens):
        pos_embeds = self.pos_weight[:tokens.shape[1], :]
        return pos_embeds.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_size)
        self.attention = nn.MultiheadAttention(cfg.embed_size, cfg.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.embed_size)
        self.fc = nn.Sequential(
            nn.Linear(cfg.embed_size, cfg.fc_size),
            nn.GELU(),
            nn.Linear(cfg.fc_size, cfg.embed_size)
        )

        

    def forward(self, X_in, padding_mask=None):
        normal_X_in = self.ln1(X_in)
        mask = nn.Transformer.generate_square_subsequent_mask(sz=X_in.shape[1]).bool().to(X_in.device)
        attn, _ = self.attention(normal_X_in, normal_X_in, normal_X_in, need_weights=False,
                              attn_mask=mask, is_causal=True, 
                              key_padding_mask=padding_mask)
        X_mid = attn + X_in
        normal_X_mid = self.ln2(X_mid)
        fc_X_mid = self.fc(normal_X_mid)
        X_out = fc_X_mid + X_mid
        return X_out
        

class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.len_vocab, cfg.embed_size, padding_idx=cfg.padding_idx)
        self.pos_embed = PositionalEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_blocks)] )
        self.ln_final = nn.LayerNorm(cfg.embed_size)
        self.fc_final = nn.Linear(cfg.embed_size, cfg.len_vocab)
        init_module(self.pos_embed, cfg)
        init_module(self.blocks, cfg)
        init_module(self.fc_final, cfg)
        self.pad_id = cfg.padding_idx
    
    def forward(self, tokens):
        padding_mask = (tokens==self.pad_id).to(tokens.device)
        embed = self.emb(tokens)
        pos_embed = self.pos_embed(tokens)
        f_embed = embed + pos_embed
        blocks_out = f_embed
        for block in self.blocks:
            blocks_out = block(blocks_out, padding_mask)
        norm_blocks_out = self.ln_final(blocks_out)
        logits = self.fc_final(norm_blocks_out)
        return logits

    @torch.no_grad()
    def inference(self, encoder, text_prefix="", temperature=1.0, max_len = 300):
        self.eval()
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        tokens = encoder.encode(text_prefix)
        tokens = [1] + tokens
        tokens = torch.LongTensor(tokens).to(device)
        new_token = -1
        while tokens.shape[0] < max_len:
            if new_token == 2 or new_token == 3:
                break
            logits = self.forward(tokens.unsqueeze(0)).squeeze(0)
            logits /= temperature
            
            new_token = torch.distributions.Categorical(logits=logits[-1, :]).sample()
            tokens = torch.cat([tokens, new_token.reshape(1)])
            new_token = new_token.item()
        
        tokens = tokens[1:]

        if tokens[-1] == 2:
            tokens = tokens[:-1]

        text = encoder.decode(tokens.tolist())
        return text

        
        




