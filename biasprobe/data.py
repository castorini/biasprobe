from typing import List, Any, Tuple, TypeVar, Callable

import numpy as np
import torch
import torch.utils.data as tud
from permsc import RankingExample

from biasprobe.serializable import TorchSerializable


__all__ = ['OrderedTensorDataset']

T = TypeVar('T')


class OrderedTensorDataset(TorchSerializable, tud.Dataset):
    def __init__(self, tensors, labels=None, original_examples=None, metadata_list=None):
        # type: (List[torch.Tensor], List[torch.Tensor], List[RankingExample], List[Any]) -> None
        self.original_examples = original_examples
        self.labels = labels
        self.tensors = tensors
        self.metadata_list = metadata_list

    def to_embeddings(self) -> torch.Tensor:
        s0 = self.tensors[0].shape[0]
        return torch.stack([x[l - 1] for x, l in zip(self.tensors, self.labels) if x.shape[0] == s0])

    def to_strings(self) -> np.ndarray:
        s0 = self.tensors[0].shape[0]
        strings = [np.array([x.content for x in x.hits])[l - 1] for x, l in zip(self.original_examples, self.labels) if len(x.hits) == s0 and len(l) == s0]
        return np.vstack(strings)

    def sliced(self, start: int = 0, end: int = None) -> 'OrderedTensorDataset':
        return OrderedTensorDataset(
            self.tensors[start:end],
            None if self.labels is None else self.labels[start:end],
            None if self.original_examples is None else self.original_examples[start:end],
            None if self.metadata_list is None else self.metadata_list[start:end],
        )

    def shuffle(self):
        shuf_idxs = np.random.permutation(len(self))
        self.tensors = np.array(self.tensors, dtype=object)[shuf_idxs].tolist()
        self.tensors = [torch.Tensor(x) for x in self.tensors]

        if self.metadata_list is not None:
            self.metadata_list = np.array(self.metadata_list, dtype=object)[shuf_idxs].tolist()

        if self.original_examples is not None:
            self.original_examples = np.array(self.original_examples, dtype=object)[shuf_idxs].tolist()

        if self.labels is not None:
            self.labels = np.array(self.labels, dtype=object)[shuf_idxs].tolist()

    def collate_fn(self, input: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.labels is None:
            labels = torch.stack([torch.arange(len(x[0])) for x in input])
        else:
            labels = torch.stack([x[1] for x in input])

        return torch.stack([x[0] for x in input]), labels

    def __getitem__(self, idx: int | slice) -> Tuple[torch.Tensor, torch.Tensor | None] | 'OrderedTensorDataset':
        if isinstance(idx, int):
            return self.tensors[idx], None if self.labels is None else self.labels[idx]
        elif isinstance(idx, slice):
            return self.sliced(idx.start or 0, idx.stop)

    def __len__(self) -> int:
        return len(self.tensors)
