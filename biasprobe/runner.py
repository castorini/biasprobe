import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Type, TypeVar

import numpy as np
import pandas as pd
import torch
from permsc import Message, RankingExample, RankingPromptBuilder, RankingDataset, Item
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import ModelOutput

from .extract import RankingPromptExtractor, LlamaPromptExtractor, LlamaExtractor, ExtractedWordEmbeddings
from .data import OrderedTensorDataset
from .serializable import TorchSerializable
from .prompt import PromptFormatter, LlamaPromptFormatter

__all__ = ['ListwiseExtractionRunner', 'ExtractionExperiment', 'PairwiseExtractionRunner', 'generate_layers']


T = TypeVar('T')


class ExtractionRunner:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def encode(self, dialog: List[Message]) -> torch.Tensor:
        return self.tokenizer.apply_chat_template([x.dict() for x in dialog], return_tensors='pt').to(self.device)

    def generate(self, dialog: List[Message], **kwargs) -> ModelOutput:
        prompt = self.encode(dialog)
        kwargs.update(dict(do_sample=False, output_hidden_states=True, return_dict_in_generate=True))

        with torch.no_grad():
            return self.model.generate(prompt, **kwargs)

    @classmethod
    def from_pretrained(cls: Type[T], model_name: str, optimize: bool = True, device_map='auto', **kwargs) -> T:
        kwargs.update(dict(trust_remote_code=True))
        kwargs.update(dict(use_flash_attention_2=True, torch_dtype=torch.float16) if optimize else {})
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return cls(model, tokenizer)

    @classmethod
    def from_config(cls: Type[T], name: str, config: AutoConfig, device_map='auto', **kwargs) -> T:
        model = AutoModelForCausalLM.from_pretrained(name, config=config, device_map=device_map, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(name)

        return cls(model, tokenizer)


def generate_layers(num_layers: int = 32, step: int = 5, include_layers: List[int] = None) -> List[int]:
    include_layers = include_layers or []
    return list(dict.fromkeys([0, 1] + list(range(step, num_layers, step)) + [num_layers] + include_layers))


class ListwiseExtractionRunner(ExtractionRunner):
    def run_extraction(
            self,
            dataset: RankingDataset,
            num_repeat: int = 1,
            max_items_per_chunk: int = 10,
            layers: List[int] = (0, 1, 5, 10, 15, 20, 25, 30, 32),
            run_inference: bool = False,
            builder: RankingPromptBuilder = None,
            prompt_extractor: RankingPromptExtractor = None,
            formatter: PromptFormatter = None,
            debug: bool = False
    ) -> 'ExtractionExperiment':
        experiment = ExtractionExperiment()
        kwargs = dict(
            layers=layers,
            run_inference=run_inference,
            builder=builder,
            prompt_extractor=prompt_extractor,
            formatter=formatter,
            debug=debug
        )

        for _ in trange(num_repeat, position=1):
            for ex in tqdm(dataset):
                ex.randomize_order()
                words = [x.content for x in ex.hits]

                if len(words) > max_items_per_chunk:
                    for split_ex in ex.split(max_items_per_chunk):
                        experiment += self.extract_example(split_ex, **kwargs)
                else:
                    experiment += self.extract_example(ex, **kwargs)

        return experiment

    def extract_example(
            self,
            example: RankingExample,
            layers: List[int] = (0, 1, 5, 10, 15, 20, 25, 30, 32),
            run_inference: bool = False,
            builder: RankingPromptBuilder = None,
            prompt_extractor: RankingPromptExtractor = None,
            formatter: PromptFormatter = None,
            debug: bool = False
    ) -> 'ExtractionExperiment':
        max_new_tokens = 300 if run_inference else 1
        prompt_extractor = prompt_extractor or LlamaPromptExtractor(model=self.tokenizer.name_or_path)
        formatter = formatter or LlamaPromptFormatter(self.tokenizer)

        items = [x.content for x in example.hits]
        true_ranking = np.argsort(example.current_permutation)

        # Do generation
        if builder is None:
            content = example.hits[0].metadata['prompt'].rstrip() + ' ' + ', '.join(items)
            content = [Message(role='user', content=content)]
            out = self.generate(content, max_new_tokens=max_new_tokens)
        else:
            content = builder.make_prompt(example)
            out = self.generate(content, max_new_tokens=max_new_tokens)

        # Extract ranking
        output_string = self.tokenizer.decode(out.sequences[0])
        embeddings = defaultdict(dict)
        pred_ranking = [prompt_extractor.extract(output_string, items)]
        true_ranking = [true_ranking + 1]
        out_seq_cpu = out.sequences[0].cpu().numpy()

        if debug:
            print(output_string)
            print(pred_ranking[-1])

        for word in items:
            pos = formatter.substr_to_positions(out_seq_cpu, word)

            for layer in layers:
                extractor = LlamaExtractor(layers=[layer], positions=pos)
                embeddings[layer][word] = extractor.extract(out).values[-1]

        # Build experiment
        extracted_embeds = {}

        for layer in layers:
            extracted_embeds[layer] = ExtractedWordEmbeddings(embeddings[layer])

        return ExtractionExperiment([extracted_embeds], pred_ranking if run_inference else [], true_ranking, [deepcopy(example)])


class PairwiseExtractionRunner(ExtractionRunner):
    def run_extraction(
            self,
            *datasets: List[str],
            num_repeat: int = 1,
            layers: List[int] = (0, 1, 5, 10, 15, 20, 25, 30, 32),
            run_inference: bool = False,
            builder: RankingPromptBuilder = None,
            prompt_extractor: RankingPromptExtractor = None,
            formatter: PromptFormatter = None,
            skip_if_found: bool = False,
            skip_if_not_found: bool = False,
            debug: bool = False,
            parallel: bool = False,
            max_new_tokens: int = None
    ) -> 'ExtractionExperiment':
        experiment = ExtractionExperiment()
        kwargs = dict(
            layers=layers,
            run_inference=run_inference,
            builder=builder,
            prompt_extractor=prompt_extractor,
            formatter=formatter,
            debug=debug,
            skip_if_found=skip_if_found,
            skip_if_not_found=skip_if_not_found,
            max_new_tokens=max_new_tokens
        )
        datasets = [pd.DataFrame(dict(words=dataset)).sample(frac=1.0, replace=False) for dataset in datasets]

        for _ in trange(num_repeat):
            if parallel:
                ds = datasets[0]
                line = str(ds.sample(1).iloc[0].words)
                words = line.split(' vs. ')
                words = [x.lower().strip() for x in words]
                a = random.randint(0, len(words) - 2)
                b = random.randint(a + 1, len(words) - 1)

                word1 = words[a]
                word2 = words[b]
            else:
                a = random.randint(0, len(datasets) - 2)
                b = random.randint(a + 1, len(datasets) - 1)
                dataset1 = datasets[a]
                dataset2 = datasets[b]

                word1 = str(dataset1.sample(1).iloc[0].words)
                word2 = str(dataset2.sample(1).iloc[0].words)

            label = 1  # word2 is bigger

            if random.random() < 0.5:
                word1, word2 = word2, word1
                label = 0  # word1 is bigger

            try:
                experiment += self.extract_example(word1, word2, label, **kwargs)
            except ValueError:
                print('Failed to find an answer OR found an answer when inappropriate, skipping...')
                continue
            except RuntimeError:
                print('Could not extract anything, skipping...')
                continue

        return experiment

    def extract_example(
            self,
            word1: str,
            word2: str,
            label: int,
            layers: List[int] = (0, 1, 5, 10, 15, 20, 25, 30, 32),
            run_inference: bool = False,
            builder: RankingPromptBuilder = None,
            prompt_extractor: RankingPromptExtractor = None,
            skip_if_found: bool = False,
            skip_if_not_found: bool = False,
            formatter: PromptFormatter = None,
            debug: bool = False,
            max_new_tokens: int = None
    ) -> 'ExtractionExperiment':
        max_new_tokens_ = 100 if run_inference else 10
        max_new_tokens = max_new_tokens or max_new_tokens_
        prompt_extractor = prompt_extractor or LlamaPromptExtractor(self.tokenizer.name_or_path)
        formatter = formatter or LlamaPromptFormatter(self.tokenizer)

        true_ranking = [0, 1] if label == 1 else [1, 0]
        true_ranking = np.array(true_ranking)
        example = RankingExample(hits=[Item(content=word1), Item(content=word2)], metadata=dict(label=label))

        # Do generation
        if builder is None:
            raise ValueError('Builder must be provided')
        else:
            content = builder.make_prompt(example)
            out = self.generate(content, max_new_tokens=max_new_tokens)

        # Extract ranking
        output_string = self.tokenizer.decode(out.sequences[0])
        embeddings = defaultdict(dict)
        pred_ranking = [prompt_extractor.extract(output_string, [word1, word2])]

        if (not prompt_extractor.last_found and skip_if_not_found) or (prompt_extractor.last_found and skip_if_found):
            raise ValueError('Prompt extractor found something or not, skipping...')

        true_ranking = [true_ranking + 1]
        out_seq_cpu = self.encode(content).cpu().numpy()[0]
        example.metadata = dict(pred_label=pred_ranking[-1][0] - 1)

        if debug:
            print(output_string)
            print(pred_ranking[-1])

        for word in (word1, word2):
            pos = formatter.substr_to_positions(out_seq_cpu, word)

            for layer in layers:
                extractor = LlamaExtractor(layers=[layer], positions=pos)
                embeddings[layer][word] = extractor.extract(out).values[-1]

        # Build experiment
        extracted_embeds = {}

        for layer in layers:
            extracted_embeds[layer] = ExtractedWordEmbeddings(embeddings[layer])

        return ExtractionExperiment([extracted_embeds], pred_ranking, true_ranking, [example])


@dataclass
class ExtractionExperiment(TorchSerializable):
    """
    A collection of extracted word/phrase embeddings, predicted labels, true labels, and original examples. The
    embeddings stored as dictionaries of layer->(word->embedding) pairs. The predicted and true labels are lists of
    the model's predicted (absent if not run) and true rankings, respectively.
    """
    embeddings_list: List[Dict[int, ExtractedWordEmbeddings]] = field(default_factory=list)
    predicted_labels: List[np.ndarray] = field(default_factory=list)
    true_labels: List[np.ndarray] = field(default_factory=list)
    original_examples: List[RankingExample] = field(default_factory=list)

    def __iadd__(self, other: 'ExtractionExperiment') -> 'ExtractionExperiment':
        self.embeddings_list += other.embeddings_list
        self.predicted_labels += other.predicted_labels
        self.true_labels += other.true_labels
        self.original_examples += other.original_examples

        return self

    def make_dataset(self, layer: int, label_type: Literal['abc', 'true', 'predicted', 'natural', 'random', 'num'] = 'natural'):
        extracted_embeddings = [x[layer] for x in self.embeddings_list]
        tensors = []
        labels = []

        for idx, embedding in enumerate(extracted_embeddings):
            tensors.append(embedding.to_tensor())
            items = embedding.items()

            match label_type:
                case 'num':
                    labels.append(np.argsort([int(x) for x, _ in items])[::-1])
                case 'abc':
                    labels.append(np.argsort([x.lower() for x, _ in items])[::-1])
                case 'true':
                    labels.append(self.true_labels[idx][::-1])
                case 'predicted':
                    labels.append(np.argsort(self.predicted_labels[idx]) + 1)
                case 'natural':
                    labels.append(np.arange(len(items)))
                case 'random':
                    labels.append(np.random.permutation(len(items)))
                case _:
                    raise ValueError(f'Unknown label type {label_type}')

        return OrderedTensorDataset(tensors, labels, self.original_examples)
