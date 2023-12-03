from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_size: int = 1000
    len_vocab: int = 5000
    padding_idx: int = 3
    max_len: int = 1024
    init_s: float = 0.02
    max_tok: int = 512
    num_heads: int = 8
    fc_size: int = 1024
    n_blocks: int = 5
