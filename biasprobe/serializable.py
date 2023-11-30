from typing import TypeVar, Type

import torch

__all__ = ['Serializable', 'TorchSerializable']

T = TypeVar('T')


class Serializable:
    @classmethod
    def load(cls: Type[T], path: str) -> T:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError


class TorchSerializable(Serializable):
    @classmethod
    def load(cls: Type[T], path: str) -> T:
        return torch.load(path)

    def save(self, path: str):
        torch.save(self, path)
