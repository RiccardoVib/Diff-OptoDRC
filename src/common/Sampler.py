# SPDX-FileCopyrightText: Copyright © 2026 Riccardo Simionato


from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler
import torch


class SequentialWithinRecordingBatchSampler(BatchSampler):
    """
    BatchSampler for datasets made of recordings split into segments.

    Expected dataset field:
        dataset.example_to_indices = {
            0: [idx_ex0_seg0, idx_ex0_seg1, idx_ex0_seg2, ...],
            1: [idx_ex1_seg0, idx_ex1_seg1, idx_ex1_seg2, ...],
            ...
        }

    Behavior:
    - choose groups of recordings of size batch_size
    - for each group, emit:
        [rec_a_seg0, rec_b_seg0, rec_c_seg0, ...]
        [rec_a_seg1, rec_b_seg1, rec_c_seg1, ...]
        ...
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = False,
    ):
        if not hasattr(dataset, "example_to_indices"):
            raise ValueError("Dataset must expose `example_to_indices`.")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.example_to_indices = dataset.example_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.example_ids = list(self.example_to_indices.keys())

    def __iter__(self):
        example_ids = self.example_ids.copy()

        if self.shuffle:
            perm = torch.randperm(len(example_ids)).tolist()
            example_ids = [example_ids[i] for i in perm]

        # Split recordings into groups of size batch_size
        groups = [
            example_ids[i:i + self.batch_size]
            for i in range(0, len(example_ids), self.batch_size)
        ]

        for group in groups:
            max_steps = max(len(self.example_to_indices[ex]) for ex in group)

            for step in range(max_steps):
                batch = []
                for ex in group:
                    seq = self.example_to_indices[ex]
                    if step < len(seq):
                        batch.append(seq[step])

                if len(batch) > 0:
                    yield batch

    def __len__(self):
        example_ids = self.example_ids.copy()
        groups = [
            example_ids[i:i + self.batch_size]
            for i in range(0, len(example_ids), self.batch_size)
        ]

        total = 0
        for group in groups:
            lengths = [len(self.example_to_indices[ex]) for ex in group]
            total += max(lengths)

        return total