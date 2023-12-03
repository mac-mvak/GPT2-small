import pickle
from tqdm import tqdm
import torch
import os


ans = []
starts = []
pkls_path = 'pkls'

lens = 0
for i, filename in enumerate(os.listdir(pkls_path)):
    with open(pkls_path + '/' + filename, mode='rb') as f:
        data = pickle.load(f)
    for tok in tqdm(data, desc=f'{i}'):
        tok1 = tok
        len1 = len(tok1)
        tok1 = torch.ShortTensor(tok1)
        start = [lens, lens + len1]
        starts.append(start)
        lens += len1
        ans.append(tok1)
    
    

ans = torch.cat(ans)
torch.save(ans, 'data/dataset.pt')
starts = torch.LongTensor(starts)
torch.save(starts, 'data/starts.pt')


