import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Any

import editdistance
import numpy as np
import torch
from permsc import RankingExample, Item, OpenAIConfig, ChatCompletionPool
from transformers.utils import ModelOutput

from .serializable import TorchSerializable

__all__ = ['ExtractedTensorDict', 'HiddenStateExtractor', 'LlamaExtractor', 'ExtractedWordEmbeddings',
           'LlamaPromptExtractor', 'RankingPromptExtractor', 'OpenAIPromptExtractor']


@dataclass
class ExtractedWordEmbeddings(TorchSerializable):
    tensor_dict: Dict[str, torch.Tensor]  # word/phrase, representation

    def to_tensor(self) -> torch.Tensor:
        return torch.stack(list(self.tensor_dict.values()))

    def items(self) -> List[tuple[str, torch.Tensor]]:
        return list(self.tensor_dict.items())

    def to_ranking_example(self) -> 'RankingExample':
        return RankingExample(hits=[Item(content=x, metadata=dict(embedding=y)) for x, y in self.tensor_dict.items()])


@dataclass
class ExtractedTensorDict(TorchSerializable):
    split: str
    tensor_dict: Dict[int, Dict[int, torch.Tensor]]  # layer, position, representation

    def get(self, layers: Set[int] | int = None, positions: Set[int] | int = None) -> torch.Tensor | Dict[int, torch.Tensor]:
        if layers is None and positions is None:
            raise ValueError('One of layer or position should be specified')

        if isinstance(layers, int):
            layers = {layers}
        elif not layers:
            layers = set()

        if isinstance(positions, int):
            positions = {positions}
        elif not positions:
            positions = set()

        filt_dict = {k: v for k, v in self.tensor_dict.items() if k in layers} if layers else self.tensor_dict
        filt_dict = {k: {p: h for p, h in v.items() if p in positions} if positions else v for k, v in filt_dict.items()}

        return filt_dict

    @property
    def values(self) -> torch.Tensor:
        return torch.stack([v2 for v1 in self.tensor_dict.values() for v2 in v1.values()])

    def state_dict(self) -> Dict[str, Any]:
        return dict(split=self.split, tensor_dict=self.tensor_dict)


class RankingPromptExtractor:
    def extract(self, prompt: str, words: List[str]) -> np.ndarray:
        raise NotImplementedError


class OpenAIPromptExtractor:
    def __init__(self, configs: List[OpenAIConfig]):
        self.pool = ChatCompletionPool(configs)

    def extract(self, prompt: str, words: List[str]) -> np.ndarray:
        content = 'Return the ranking of the words "' + ', '.join(words) + '" in the output string "' + prompt + '"'
        response = self.pool.create_batch(messages=[[dict(role='user', content=content)]], temperature=0, request_timeout=5)
        print(words, prompt, response)


class LlamaPromptExtractor(RankingPromptExtractor):
    def __init__(self, model: str = 'llama'):
        self.model = model
        self.last_found = False

    def extract(self, prompt: str, words: List[str]) -> np.ndarray:
        words = [x.lower().strip() for x in words]
        prompt = prompt.lower()
        prompt = prompt[prompt.find('/inst]') + 6:]

        if 'vicuna' in self.model.lower():  # Vicuna
            prompt = prompt[prompt.find('\n') + 1:]

        prompt = prompt.replace('</s>', '').strip()
        prompt = re.sub(r'\b(positive|negative|bad|good|moral|immoral):', '', prompt, re.IGNORECASE).lower().strip()
        rankings = []

        for word in words:
            word = word.lower()

            if (idx := prompt.find(word)) != -1:
                rankings.append(idx)
                self.last_found = True
            else:
                rankings.append(10000)

        for x in ('neither', 'both', 'either', 'cannot', 'more negative'):
            if x in prompt:
                self.last_found = False

        if np.all(np.array(rankings) == 10000):
            self.last_found = False

        rankings = np.array(rankings)
        idxs = np.argsort(rankings)
        rankings = np.empty_like(rankings)
        rankings[idxs] = np.arange(len(rankings)) + 1

        return rankings


class HiddenStateExtractor:
    def __init__(self, layers=None, positions=None, split='input'):
        # type: (List[int], List[int], str) -> None
        self.layers = layers or []
        self.positions = positions or []
        self.split = split

    def extract(self, model_output: ModelOutput) -> ExtractedTensorDict:
        raise NotImplementedError


class LlamaExtractor(HiddenStateExtractor):
    def extract(self, model_output: ModelOutput) -> ExtractedTensorDict:
        hid_states = model_output['hidden_states']
        tensor_dict = defaultdict(lambda: defaultdict(dict))

        if self.split == 'output':
            for pos in self.positions:
                for layer in self.layers:
                    tensor_dict[layer][pos] = hid_states[pos][layer][0, 0]
        else:  # 'input'
            for pos in self.positions:
                for layer in self.layers:
                    tensor_dict[layer][pos] = hid_states[0][layer][0, pos]

        return ExtractedTensorDict(self.split, {k1: {k2: v2 for k2, v2 in v1.items()} for k1, v1 in tensor_dict.items()})
