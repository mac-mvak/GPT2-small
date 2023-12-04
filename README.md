## How to run Model.

First of all download TinyStories dataset to `files` directory. Then move `data49.json` to `test_data`.

After that build tokenizer using following scripts.

```
python3 preprocessing/create_tokenizer.py
python3 preprocessing/tokenizer.py
python3 preprocessing/concatter.py
```


Then run `main.py` using one of the configs in configs directory


In order to generate files use `tester.py` and `gpt2_xl.py`. In order to count perplexity 
use `evaluate.py`

