import sentencepiece
import json
import torch
from src.model import GPT2
from tqdm import tqdm
import pickle
import re


encoder = sentencepiece.SentencePieceProcessor(model_file='tex.model')


text = 'Once upon a time, in a small town, there was a kind teacher named Miss Lily. She loved to teach little kids.'


device = torch.device('cuda:0')

scaler = torch.cuda.amp.GradScaler(enabled=True)

checkpoint = torch.load('checkpoints/big_checkpoint_0.pth')

model = GPT2(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['state_dict'])


with open('test_data/data49.json') as f:
    data_f = json.load(f)


texts = []

temperature = 0.5
model_name = 'second_model'

for st in tqdm(data_f[:50]):
    story = st['story'].strip()
    story = story.split('.')
    crop_story = ".".join(story[:2])
    gen_story = model.inference(encoder, text_prefix=crop_story, temperature=temperature, max_len=500)
    texts.append(gen_story)

with open(f'generated/{model_name}_t={temperature}.pkl', 'wb') as f:
    pickle.dump(texts, f)
