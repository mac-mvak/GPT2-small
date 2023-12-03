import json
import sentencepiece
from tqdm import tqdm
import os

file_path = 'files'
txt_path = 'data/dataset.txt'
ans = []
texts = ''


if os.path.exists(txt_path):
    os.remove(txt_path)

for filename in tqdm(os.listdir(file_path)):
    with open(file_path + '/' + filename) as f:
        data = json.load(f)
    
    with open(txt_path, 'a') as f:
        for text in data:
            f.write(text['story'].strip('\n ') + '\n')



sentencepiece.SentencePieceTrainer.train(input=txt_path, model_prefix='tex', model_type="bpe",
                                        vocab_size=5000, 
                                        unk_id=0, bos_id=1, eos_id=2, pad_id=3)




