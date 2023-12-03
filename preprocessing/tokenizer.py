import json
import sentencepiece
import os
from tqdm import tqdm
import pickle


encoder = sentencepiece.SentencePieceProcessor(model_file='tex.model')
file_path = 'files'
ans = []
print(encoder.bos_id(), encoder.unk_id(), encoder.eos_id(), encoder.pad_id())


for i, filename in enumerate(os.listdir(file_path)):
    ans = []
    with open(file_path + '/' + filename) as f:
        data = json.load(f)
    
    for text in tqdm(data, desc=f'{i}'):
        ans.append(encoder.encode(text['story'].strip('\n ')))
        #f.write(text['story'].strip('\n ') + '\n')
    
    with open(f'pkls/{i}.pkl', 'wb') as f:
        pickle.dump(ans, f)
