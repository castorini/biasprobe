import hashlib
from typing import Iterable, List, Tuple, Literal

import numpy as np
from permsc import KemenyOptimalAggregator
from python_tsp.exact import solve_tsp_dynamic_programming
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoConfig

from biasprobe import OrderedTensorDataset

__all__ = ['TransEDistanceProbe', 'Probe', 'ProbeConfig', 'KernelProbe', 'PairScoreObjective',
           'SingleScoreObjective', 'VectorProbe', 'UnsupervisedTSPProbe', 'VectorProbe', 'BallProbe', 'ProbeTrainer',
           'BinaryProbe', 'WeatProbe', 'BinaryScoreObjective', 'LogisticRegressionProbe']


class ProbeConfig(BaseModel):
    metric: str = 'l2'
    positive_definite: bool = True
    input_format: str = 'BLH'  # B for batch, L for length, H for hidden size
    outer_hidden_dim: int = 4096
    inner_hidden_dim: int = 2
    use_linear_projection: bool = True

    @classmethod
    def create_for_model(cls, model: str, **kwargs) -> 'ProbeConfig':
        return cls(outer_hidden_dim=AutoConfig.from_pretrained(model).hidden_size, **kwargs)


class Probe(nn.Module):
    def __init__(self, config: ProbeConfig = None):
        super().__init__()
        self.config = config = config or ProbeConfig()
        self.dims = {k: v for v, k in enumerate(config.input_format)}

    @property
    def method(self):
        return 'single'

    @property
    def do_train(self) -> bool:
        return True

    def _swap_hidden_and_last(self, x: torch.Tensor) -> torch.Tensor:
        perm = torch.arange(x.dim())
        perm[self.dims['H']] = perm[-1]
        perm[-1] = self.dims['H']

        return x.permute(*perm)

    def split_lengthwise(self, x: torch.Tensor) -> Iterable[torch.Tensor]:
        for split_x in x.split(1, self.dims['L']):
            yield split_x

    def duplicate_lengthwise(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x = x.unsqueeze(dim)
        expand_in = [-1] * x.dim()
        expand_in[dim] = x.shape[self.dims['L']]

        return x.expand(*expand_in)

    def permute(self, x: torch.Tensor, named_dim: str, dim: int = -1) -> torch.Tensor:
        perm = torch.arange(x.dim())
        perm[self.dims[named_dim]] = perm[dim]
        perm[dim] = self.dims[named_dim]

        return x.permute(*perm)

    def _maybe_to_positive_definite(self, X: torch.Tensor) -> torch.Tensor:
        return X.T @ X if self.config.positive_definite else X  # approximate


class ProbeTrainer:
    def __init__(self, probe: Probe, objective: nn.Module = None):
        self.probe = probe
        self.method = probe.method

        if objective is None:
            if self.method == 'single':
                self.objective = SingleScoreObjective()
            elif self.method == 'pair':
                self.objective = PairScoreObjective()
            elif self.method == 'binary':
                self.objective = BinaryScoreObjective()
            else:
                self.objective = None
        else:
            self.objective = objective

    def fit(self, dataset: OrderedTensorDataset, num_epochs: int = 3, lr=5e-4, weight_decay=0.01):
        if not self.probe.do_train:
            return

        optimizer = AdamW(self.probe.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in trange(num_epochs):
            for idx, (tensor, labels) in enumerate(dataset):
                if self.objective is None:
                    continue

                optimizer.zero_grad()
                labels = labels - 1  # 1-indexed
                tensor = tensor.unsqueeze(0).cuda().float()

                if self.method == 'binary':
                    if tensor.shape[1] != 2:
                        continue

                    tensor = tensor[:, labels]

                x = self.probe(tensor).squeeze()
                loss = self.objective(x[labels] if x.dim() >= 1 else x)
                loss.backward()
                optimizer.step()

    def predict(self, dataset: OrderedTensorDataset, eval_fn: Literal['spearmanr', 'binary'] = 'spearmanr') -> Tuple[float, List[List[str]]]:
        self.probe.eval()
        rank_fn = lambda x, y: spearmanr(x, y)[0] if eval_fn == 'spearmanr' else int(x[0] == y[0])
        rhos = []
        preds_list = []

        for idx, (tensor, labels) in enumerate(dataset):
            with torch.no_grad():
                labels = labels - 1  # 1-indexed
                n = tensor.shape[0]
                z = np.random.permutation(n)

                try:
                    x = self.probe(tensor[labels][z].unsqueeze(0).cuda().float()).squeeze()
                except IndexError:
                    continue

                if self.probe.method == 'pair':
                    z = np.random.permutation(n)
                    x = x[z, :]
                    x = x[:, z]
                    pred = self.probe.decode(x)
                    pred = [z[x] for x in pred]
                elif self.probe.method == 'binary':
                    pred = [0, 1] if x.item() > 0 else [1, 0]
                    pred = np.array(pred)
                    pred = z[pred]
                else:
                    pred = np.argsort(x.cpu().numpy())
                    pred = z[pred]

            if dataset.original_examples is not None:
                items = [x.content for x in dataset.original_examples[idx].hits]
                preds_list.append(np.array(items, dtype=object)[labels][pred].tolist())

            rhos.append(rank_fn(pred, np.arange(len(pred))))

        return np.mean(rhos), preds_list


class TransEDistanceProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        h_inner = self.config.inner_hidden_dim
        self.lin_proj = nn.Linear(h_outer, h_inner, bias=False)
        self.b = nn.Parameter(torch.zeros(h_inner), requires_grad=True)

    @property
    def method(self):
        return 'pair'

    def forward(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)
        scores = self._swap_hidden_and_last(self.lin_proj(x))

        scores_i = self.duplicate_lengthwise(scores)
        scores_j = self.duplicate_lengthwise(scores)
        scores_j = self.permute(scores_j, 'L', -1)
        delta = scores_i - scores_j + self.b

        return delta.norm(p=2, dim=self.dims['H'])  # distances


class KernelProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        h_inner = self.config.inner_hidden_dim
        self.lin_proj = nn.Linear(h_outer, h_inner, bias=False)
        self.r = nn.Parameter(torch.zeros(h_outer), requires_grad=True)

    @property
    def method(self):
        return 'pair'

    def decode(self, matrix: torch.Tensor) -> np.ndarray:
        graph = (matrix > matrix.T) * (matrix - matrix.T)
        return KemenyOptimalAggregator().solve_from_graph(graph.cpu().numpy().T)

    def forward(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)
        scores = self._swap_hidden_and_last(self.lin_proj(x))

        h_i = self.duplicate_lengthwise(scores)
        h_j = self.duplicate_lengthwise(scores)
        h_j = self.permute(h_j, 'L', -1)
        delta = h_i - h_j + self.lin_proj(self.r).unsqueeze(-1)
        delta = (delta ** 2).sum(dim=self.dims['H'])

        return delta / delta.norm(p=2)


class UnsupervisedTSPProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)

    def forward(self, scores: torch.Tensor):
        scores_i = self.duplicate_lengthwise(scores)
        scores_j = self.duplicate_lengthwise(scores)
        scores_j = self.permute(scores_j, 'L', -1)

        delta = 1 - F.cosine_similarity(scores_i, scores_j, dim=self.dims['H'])
        Ds = delta.cpu().numpy()  # distances
        paths = []
        n = Ds.shape[-1]

        fwd_shift_perm = [n - 1] + list(range(n - 1))
        fwd_shift_perm = np.array(fwd_shift_perm)

        for D in Ds:
            min_cost = 10000000
            min_path = None
            offset = 0

            rand_perm = np.random.permutation(n)
            D = D[rand_perm, :]
            D = D[:, rand_perm]
            restore_map = dict(zip(range(n), rand_perm))

            for i in range(n):
                D2 = np.copy(D)
                D2[:, 0] = 0
                path, cost = solve_tsp_dynamic_programming(D2)
                path = np.array(path)
                path = (path - offset) % len(path)

                if cost < min_cost:
                    min_path, min_cost = path, cost

                D = D[fwd_shift_perm, :]
                D = D[:, fwd_shift_perm]
                offset += 1

            paths.append(np.array([restore_map[x] for x in min_path]))

        return paths


class BallProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        h_inner = self.config.inner_hidden_dim
        self.r = nn.Parameter(torch.zeros(h_outer), requires_grad=True)
        nn.init.normal_(self.r.data)
        self.X = nn.Linear(h_outer, h_inner, bias=False).weight
        self.lin = nn.Linear(h_outer, 1, bias=False)
        self.mean = nn.Parameter(torch.zeros(h_outer), requires_grad=False)
        self.n = 0

    def project(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)
        x -= self.mean.unsqueeze(0)

        if self.config.metric == 'l2':
            r = self.r.unsqueeze(0)

            if self.config.use_linear_projection:
                x = F.linear(x, self.X)
                r = F.linear(r, self.X)

            return x, r
        elif self.config.metric == 'cos':
            r = self.r.unsqueeze(0)

            if self.config.use_linear_projection:
                x = F.linear(x, self.X)
                r = F.linear(r, self.X)

            return x, r
        else:
            raise ValueError('Metric unknown')

    def forward(self, x: torch.Tensor):
        if self.r.norm(p=2).item() > 1000:
            self.r.data = self.r.data / (self.r.data.norm(p=2) / 1000)

        x = self._swap_hidden_and_last(x)

        if self.config.metric == 'l2':
            if self.config.use_linear_projection:
                if self.training:
                    self.mean.data = self.mean.data * self.n / (self.n + 1) + x.mean(dim=self.dims['L']) / (self.n + 1)

                x -= self.mean
                x = F.linear(x, self.X)
                self.n += 1

            x = self._swap_hidden_and_last(x)
            x = x.norm(p=2, dim=self.dims['H'])
        else:
            raise ValueError('Metric unknown')

        return x / x.norm(p=2)


class VectorProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        h_inner = self.config.inner_hidden_dim
        self.r = nn.Parameter(torch.zeros(h_outer), requires_grad=True)
        nn.init.normal_(self.r.data)
        self.X = nn.Linear(h_outer, h_inner, bias=False).weight
        self.lin = nn.Linear(h_outer, 1, bias=False)

    def project(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)

        match self.config.metric:
            case 'l2':
                r = self.r.unsqueeze(0)

                if self.config.use_linear_projection:
                    x = F.linear(x, self.X)
                    r = F.linear(r, self.X)

                return x, r
            case 'cos' | 'dot':
                r = self.r.unsqueeze(0)

                if self.config.use_linear_projection:
                    x = F.linear(x, self.X)
                    r = F.linear(r, self.X)

                return x, r
            case _:
                raise ValueError('Metric unknown')

    def forward(self, x: torch.Tensor):
        if self.r.norm(p=2).item() > 1000:
            self.r.data = self.r.data / (self.r.data.norm(p=2) / 1000)

        x = self._swap_hidden_and_last(x)

        if self.config.metric == 'l2':
            delta = x - self.r.unsqueeze(0)

            if self.config.use_linear_projection:
                delta = F.linear(delta, self.X)

            delta = self._swap_hidden_and_last(delta)
            delta = delta.norm(p=2, dim=self.dims['H'])
        elif self.config.metric == 'dot':
            r = self.r.unsqueeze(0)

            if self.config.use_linear_projection:
                x = F.linear(x, self.X)
                r = F.linear(r, self.X)

            delta = r.matmul(x.permute(0, 2, 1))
        elif self.config.metric == 'cos':
            r = self.r.unsqueeze(0)

            if self.config.use_linear_projection:
                x = F.linear(x, self.X)
                r = F.linear(r, self.X)

            delta = 1 - F.cosine_similarity(x, r, -1)
        else:
            raise ValueError('Metric unknown')

        return delta / (delta.norm(p=2) + 1e-10)


class WeatProbe(Probe):
    def __init__(self, word_vecs1: torch.Tensor, word_vecs2: torch.Tensor, config: ProbeConfig = None):
        super().__init__(config)
        self.word_vecs1 = word_vecs1
        self.word_vecs2 = word_vecs2

    @property
    def method(self):
        return 'binary'

    @property
    def do_train(self):
        return False

    @classmethod
    def from_dataset(cls, dataset: 'OrderedTensorDataset', hash_fn=None, **hash_kwargs) -> Tuple['WeatProbe', 'OrderedTensorDataset', 'OrderedTensorDataset']:
        def string_hash_fn(string, pct=80):
            return (int(hashlib.md5(string.encode()).hexdigest(), 16) % 100) < pct

        hash_fn = hash_fn or string_hash_fn
        ds = dataset

        a_strings = np.reshape(ds.to_strings(), (-1,))[::2]
        b_strings = np.reshape(ds.to_strings(), (-1,))[1::2]
        a_embeds = ds.to_embeddings().permute(1, 0, 2)[0]
        b_embeds = ds.to_embeddings().permute(1, 0, 2)[1]

        a_train_idxs = np.array([hash_fn(x, **hash_kwargs) for x in a_strings])
        b_train_idxs = np.array([hash_fn(x, **hash_kwargs) for x in b_strings])
        a_train_idxs = a_train_idxs[:len(a_embeds)]
        b_train_idxs = b_train_idxs[:len(b_embeds)]

        train_e_a = a_embeds[a_train_idxs]
        train_e_b = b_embeds[b_train_idxs]
        test_e_a = a_embeds[~a_train_idxs]
        test_e_b = b_embeds[~b_train_idxs]

        min_train = min(len(train_e_a), len(train_e_b))
        min_test = min(len(test_e_a), len(test_e_b))

        train_new_e = torch.stack((train_e_a[:min_train], train_e_b[:min_train]), 1)
        test_new_e = torch.stack((test_e_a[:min_test], test_e_b[:min_test]), 1)

        new_train_ds = OrderedTensorDataset(train_new_e, torch.tensor([1, 2]).unsqueeze(0).repeat(len(train_new_e), 1).cpu().numpy())
        new_test_ds = OrderedTensorDataset(test_new_e, torch.tensor([1, 2]).unsqueeze(0).repeat(len(test_new_e), 1).cpu().numpy())

        return cls(train_e_a, train_e_b), new_train_ds, new_test_ds

    def mean_cosine_sim(self, x: torch.Tensor, vecs: torch.Tensor):
        return F.cosine_similarity(x.repeat(vecs.shape[0], 1), vecs, -1).mean(dim=0)

    def forward(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)
        mcs11 = self.mean_cosine_sim(x[:, 0].to(self.word_vecs1.device), self.word_vecs1)
        mcs12 = self.mean_cosine_sim(x[:, 0].to(self.word_vecs1.device), self.word_vecs2)
        mcs21 = self.mean_cosine_sim(x[:, 1].to(self.word_vecs1.device), self.word_vecs1)
        mcs22 = self.mean_cosine_sim(x[:, 1].to(self.word_vecs1.device), self.word_vecs2)

        return mcs11 - mcs12 - (mcs21 - mcs22)


class BinaryProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        self.lin = nn.Linear(h_outer, 1)

    @property
    def method(self):
        return 'binary'

    def forward(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)
        x = self.lin(x)

        return x[:, 1] - x[:, 0]


class LogisticRegressionProbe(Probe):
    def __init__(self, config: ProbeConfig = None):
        super().__init__(config)
        h_outer = self.config.outer_hidden_dim
        self.lin = nn.Linear(2 * h_outer, 1)

    @property
    def method(self):
        return 'binary'

    def forward(self, x: torch.Tensor):
        x = self._swap_hidden_and_last(x)

        return self.lin(x.reshape(x.size(0), -1))


class SingleScoreObjective(nn.Module):
    def __init__(self, use_max_margin: bool = True, c: float = 0.5):
        super().__init__()
        self.use_max_margin = use_max_margin
        self.c = c

    def forward(self, s: torch.Tensor, perm=None):
        n = s.shape[-1]
        w = torch.triu(torch.full((n, n), -1), diagonal=0).to(s.device)
        w[torch.arange(n), torch.arange(n)] = torch.arange(n - 1, -1, -1).to(s.device)
        w = w.float()

        if perm is not None:
            w = w[perm, :]
            w = w[:, perm]

        if self.use_max_margin:
            s = s.squeeze()

            if perm is not None:
                s = s[perm]

            s = s.unsqueeze(0)
            si = s.repeat(n, 1)
            sj = s.permute(1, 0).repeat(1, n)
            sij = sj - si + self.c

            return torch.relu(torch.triu(sij, diagonal=1)).mean()
        else:
            return s.matmul(w.T).mean()


class PairScoreObjective(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d: torch.Tensor, perm=None):
        # assume bsz is 1 and natural order for now
        d = d[0]
        n = d.shape[-1]
        mask = torch.triu(torch.ones(n, n), diagonal=1).to(d.device)
        mask = mask - mask.T

        return (mask * d).sum()


class BinaryScoreObjective(nn.Module):
    def __init__(self, use_max_margin: bool = True, c: float = 1):
        super().__init__()
        self.use_max_margin = use_max_margin
        self.c = c

    def forward(self, scores: torch.Tensor, perm=None):
        if self.use_max_margin:
            return -torch.relu(scores + self.c).mean()
        else:
            return F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores))
