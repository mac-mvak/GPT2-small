import json
import sentencepiece

file_path = 'files/data00.json'
ans = []
texts = ''

with open(file_path) as f:
    fs = json.load(f)

for i in range(50):
    ans.append({
        'story' : fs[i]['story']
    })
    texts += fs[i]['story']

with open('dataset.json', 'w') as f:
    json.dump(ans, f)

with open('dataset.txt', 'w') as f:
    f.write(texts)

sentencepiece.SentencePieceTrainer.train(input='./dataset.txt', model_prefix='m', vocab_size=1000, 
                                         unk_id=0, bos_id=1, eos_id=2, pad_id=3)




