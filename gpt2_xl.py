from transformers import pipeline, set_seed
import json
import pickle
from tqdm import tqdm

generator = pipeline('text-generation', model='gpt2-xl', device=0)
set_seed(42)


with open('test_data/data49.json') as f:
    data_f = json.load(f)


texts = []



for st in tqdm(data_f[:50]):
    story = st['story'].strip()
    story = story.split('.')
    crop_story = ".".join(story[:2])
    gen_story = generator(crop_story, max_length=250, num_return_sequences=1)
    gen_story = gen_story[0]['generated_text']
    texts.append(gen_story)

with open(f'generated/gpt2-xl.pkl', 'wb') as f:
    pickle.dump(texts, f)
