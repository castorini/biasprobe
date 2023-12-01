# BiasProbe: Probing What Llamas Really Think
[![HF Spaces](https://img.shields.io/badge/HuggingFace%20Space-online-green.svg)](https://huggingface.co/spaces/tetrisd/biasprobe)
[![Citation](https://img.shields.io/badge/Citation-arXiv-orange.svg)](https://github.com/castorini/biasprobe#citation)
[![PyPi version](https://badgen.net/pypi/v/biasprobe?color=blue)](https://pypi.org/project/biasprobe)
<!--[![Downloads](https://static.pepy.tech/badge/biasprobe)](https://pepy.tech/project/biasprobe)-->
![image](https://github.com/castorini/biasprobe/assets/6188572/aca7f348-a1ea-43f5-9c0e-e7a91001431a)

This repository contains the code for our paper, "[What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations](https://arxiv.org/abs/2311.18812)."

## Getting Started

1. Install the PyPI package:
```bash
pip install biasprobe
```

2. Extract some embeddings. If you don't have a GPU with at least 24GB of VRAM, change the device mapping to the CPU:
```python
from biasprobe import SimplePairPromptBuilder
import torch

# Load the LLM and extractors. `optimize=True` requires FlashAttention (`pip install flash-attn`)
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
runner = PairwiseExtractionRunner.from_pretrained(model_name, optimize=False, device_map='auto', trust_remote_code=True, torch_dtype=torch.float16)
builder = SimplePairPromptBuilder(criterion='more positive')

# Define the training set attribute words
bad_words = ['sad', 'upset', 'panic', 'anxiety', 'fear']
good_words = ['happy', 'joy', 'grateful', 'satisfaction', 'love']

# Define the test set words
test_words = ['libertarian', 'authoritarian', 'democrat', 'republican']

# Run the extraction
train_exp = runner.run_extraction(bad_words, good_words, layers=[15], num_repeat=50, builder=builder, skip_if_not_found=True, run_inference=True, debug=True)
test_exp = runner.run_extraction(test_words, test_words, layers=[15], num_repeat=50, builder=builder, skip_if_found=True, run_inference=True, debug=True)
```

3. Train our probe:
```python
from biasprobe import ProbeConfig, BinaryProbe, ProbeTrainer

train_ds = train_exp.make_dataset(15, label_type='predicted')
test_ds = test_exp.make_dataset(15)
config = ProbeConfig.create_for_model('mistralai/Mistral-7B-Instruct-v0.1')
probe = BinaryProbe(config)

trainer = ProbeTrainer(probe.cuda())
trainer.fit(train_ds)
_, preferred_pairs = trainer.predict(test_ds)
```

4. `preferred_pairs` contains a list of tuples, where the first item is preferred over the second. Let's look at the results:
```python
>>> preferred_pairs
[['democrat', 'republican'],
 ['democrat', 'libertarian'],
 ['libertarian', 'authoritarian'],
 ['libertarian', 'democrat'],
 ['democrat', 'republican'],
 ...
```
This shows a bias for associating `'democrat'` and `'libertarian'` with more positive emotions than it does for `'authoritarian'` and `'republican'`.

## Citation
```
@article{tang2023found,
  title={What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations},
  author={Tang, Raphael and Zhang, Xinyu and Lin, Jimmy and Ture, Ferhan},
  journal={arXiv:2311.18812},
  year={2023}
}
```
