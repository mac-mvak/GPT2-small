import pickle
from evaluate import load

with open('generated/first_model_t=0.8.pkl', 'rb') as f:
    preds = pickle.load(f)

perplexity = load("perplexity", module_type="metric", device=0)
results = perplexity.compute(predictions=preds, model_id='gpt2')
print(results)



