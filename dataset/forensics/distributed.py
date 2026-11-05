"""Deterministic balanced sharding for Unified Forensics data parallelism."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class BalancedDistributedForensicsSampler(Sampler[int]):
    """Shard paired Real/Fake samples without cross-rank duplication.

    Every local micro-batch of size two is one Real plus one Fake. Across an
    epoch, each source row appears on exactly one rank when both class counts
    are divisible by ``num_replicas`` (true for the frozen Phase 2A split).
    """

    def __init__(self, dataset, *, num_replicas: int, rank: int, seed: int = 0,
                 shuffle: bool = True) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid distributed sampler rank {rank}/{num_replicas}")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.real_indices = [i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 0]
        self.fake_indices = [i for i, row in enumerate(dataset.rows) if int(row["class_label"]) == 1]
        if len(self.real_indices) != len(self.fake_indices):
            raise ValueError(
                f"Phase 2A requires balanced Real/Fake rows, got "
                f"{len(self.real_indices)}/{len(self.fake_indices)}"
            )
        if len(self.real_indices) % self.num_replicas:
            raise ValueError("Class counts must be divisible by world size to avoid padding duplicates")
        self.pairs_per_rank = len(self.real_indices) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _permutation(self, values, stream: int):
        if not self.shuffle:
            return list(values)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1009 + stream)
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[index] for index in order]

    def __iter__(self):
        real = self._permutation(self.real_indices, 17)
        fake = self._permutation(self.fake_indices, 31)
        local = []
        for pair_index in range(self.pairs_per_rank):
            global_index = pair_index * self.num_replicas + self.rank
            # Rotate within-rank order to avoid a fixed domain-first trajectory
            # while retaining exactly one Real and one Fake per micro-batch.
            pair = (real[global_index], fake[global_index])
            local.extend(pair if (pair_index + self.epoch + self.rank) % 2 == 0 else pair[::-1])
        return iter(local)

    def __len__(self) -> int:
        return self.pairs_per_rank * 2


def sampler_epoch_indices(dataset, *, world_size: int, seed: int, epoch: int):
    output = []
    for rank in range(world_size):
        sampler = BalancedDistributedForensicsSampler(
            dataset, num_replicas=world_size, rank=rank, seed=seed, shuffle=True
        )
        sampler.set_epoch(epoch)
        output.append(list(iter(sampler)))
    return output
