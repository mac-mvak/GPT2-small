import os.path

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab, build_vocab_from_iterator
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from typing import Iterable, List
from timeit import default_timer as timer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sacrebleu.metrics import BLEU
from torch import Tensor
import torch
import subprocess
import wandb
import numpy as np
import torch.nn as nn
from torch.nn import Transformer
import math
from tqdm import tqdm



# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}
vocab_sizes = {
SRC_LANGUAGE: 20000,
TGT_LANGUAGE: 20000
}
train_texts = {}
val_texts = {}
with open('data/train.de-en.en') as f:
    train_texts[TGT_LANGUAGE] = f.read().splitlines()
with open('data/train.de-en.de') as f:
    train_texts[SRC_LANGUAGE] = f.read().splitlines()
with open('data/val.de-en.en') as f:
    val_texts[TGT_LANGUAGE] = f.read().splitlines()
with open('data/val.de-en.de') as f:
    val_texts[SRC_LANGUAGE] = f.read().splitlines()



for language in [SRC_LANGUAGE, TGT_LANGUAGE]: # сделаем чтобы токенизация была такая же, как в исходном тексте
    token_transform[language] = lambda sent: sent.split()



# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        u = token_transform[language](data_sample[language_index[language]])
        yield u

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = zip(train_texts[SRC_LANGUAGE], train_texts[TGT_LANGUAGE])
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    max_tokens=vocab_sizes[ln],
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.

def beam_decode(model, src, src_mask, max_len, start_symbol, beam_width = 2, alpha=0.75): # колхоз
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    y = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    ys = [[y, 0] for _ in range(beam_width)]
    ans = []
    for i in range(max_len - 1):
        best_results = []
        for y, y_prob in ys:
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(y.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(y, memory, tgt_mask)
            out = out.transpose(0, 1)
            out = out[:, -1]
            prob = model.generator(out).log_softmax(-1).flatten()
            inds = torch.argsort(prob, descending=True)[:beam_width]
            for k in range(inds.shape[0]):
                next_word = inds[k].item()
                next_prob = prob[inds[k]].item()
                new_y = torch.cat([y,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                best_results.append((new_y, y_prob + next_prob))
        best_results.sort(key=lambda x: x[1], reverse=True)
        ys = []
        for pair in best_results:
            if pair[0][-1].item() == EOS_IDX:
                ans.append(pair)
            else:
                ys.append(pair)
            if len(ans) == beam_width or len(ys) == beam_width:
                break
        if len(ans) == beam_width:
            break
    ys = ans + ys
    for k in range(len(ys)):
        y, log_prob = ys[k]
        prob = np.exp(log_prob)
        prob /= len(y)**(alpha)
    ys.sort(key=lambda x: x[1], reverse=True)
    y = ys[0][0]
    return y

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_decode(
        model,  src, src_mask, max_len=num_tokens + num_tokens//2, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = EMB_SIZE
BATCH_SIZE = 150
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.25)


class TranslatorDataset(Dataset):
    def __init__(self, src, tgt):
        self.scr = src
        self.tgt = tgt

    def __len__(self):
        return len(self.scr)

    def __getitem__(self, idx):
        return self.scr[idx], self.tgt[idx], idx


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        #txt_input = transforms[0].encode(txt_input,  out_type=str)
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch, idx = [], [], []
    for src_sample, tgt_sample, id in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
        idx.append(id)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch, idx


def check_val_bleu(text):
    f = open('tmp/translate.txt', 'w')
    for sent in text:
        f.write(sent.replace('<unk>', '') + '\n')
    f.close()
    w = float(
        subprocess.check_output(['cat tmp/translate.txt | sacrebleu data/val.de-en.en  --tokenize none --width 2 -b'], shell=True)
    )
    return w


def train_epoch(model, optimizer, text):
    model.train()
    losses = 0
    train_iter = TranslatorDataset(train_texts[SRC_LANGUAGE], train_texts[TGT_LANGUAGE])
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt, idx in tqdm(train_dataloader, desc=text):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_dataloader)


def evaluate(model, epoch, text):
    model.eval()
    losses = 0
    val_iter = TranslatorDataset(val_texts[SRC_LANGUAGE], val_texts[TGT_LANGUAGE])
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    pred_texts, ref_text = [], []

    for src, tgt, idx in tqdm(val_dataloader, desc=text):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        if epoch % 5 == 0:
            for id in idx:
                ref_text.append(val_texts[TGT_LANGUAGE][id])
                pred_texts.append(
                    translate(model, val_texts[SRC_LANGUAGE][id]))
    if epoch % 5 == 0:
        score = check_val_bleu(pred_texts)
    else:
        score = 0.
    return score, losses / len(val_dataloader)

def saver(model, epoch):
    with open('data/test1.de-en.de') as f:
        final_texts = f.read().splitlines()
    f = open(f'savings/final_text_{epoch}.txt', 'w')
    for text in tqdm(final_texts):
        text = translate(model, text)
        f.write(text.replace('<unk> ', '') + '\n')
    f.close()


wandb.login(key='4c350b2d13a9e01823b87d63216300deafe6b513')
NUM_EPOCHS = 500
wandb.init(
        project="BHW-2",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            'dict_size': SRC_LANGUAGE,
            'type': f'Default Transformer, {NUM_ENCODER_LAYERS} layers,scheduler , default hidden layer, smoothing',
            'layers': NUM_ENCODER_LAYERS
    })



for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, f'train, epoch={epoch}')
    end_time = timer()
    val_score, val_loss = evaluate(transformer, epoch, f'test, epoch={epoch}')
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']
    if epoch % 5 == 0:
        saver(transformer, epoch)
    metrics = {"train/train_loss": train_loss,
               "train/epoch": epoch,
               "val/val_loss": val_loss,
               'lr': lr
               }
    if epoch % 5 == 0:
        metrics['val/val_bleu'] = val_score
    wandb.log(metrics)


# function to generate output sequence using greedy algorithm


# actual function to translate input sentence into target language




