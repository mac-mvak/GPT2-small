import sentencepiece
import json

model = sentencepiece.SentencePieceProcessor(model_file='m.model')

with open('dataset.json') as f:
    files = json.load(f)

a = []
for i in range(len(files)):

    t1 = files[i]['story']

    w = model.encode(t1)
    a.append(len(w))

print(max(a))
#u = model.decode(w)
#print(u)