import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import config

class BinaryMemmapDataset(Dataset):
    """
    High-speed Memory-mapped Binary Dataset Loader.
    Reads pre-tokenized uint16/int32 binary files via `np.memmap` for zero CPU latency.
    """
    def __init__(self, data_dir: str = config.dataset_dir):
        meta_path = os.path.join(data_dir, "binary_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Binary metadata file not found at: {meta_path}. Run build_binary_dataset.py first!")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.total_samples = self.meta["total_samples"]
        self.bin_src_path = self.meta["bin_src_path"]
        self.bin_tgt_path = self.meta["bin_tgt_path"]
        self.bin_offsets_path = self.meta["bin_offsets_path"]

        # Memory map the binary files into RAM without copying
        self.src_mmap = np.memmap(self.bin_src_path, dtype=np.int32, mode="r")
        self.tgt_mmap = np.memmap(self.bin_tgt_path, dtype=np.int32, mode="r")
        self.offsets_mmap = np.memmap(self.bin_offsets_path, dtype=np.int64, mode="r")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        start_idx = self.offsets_mmap[idx]
        end_idx = self.offsets_mmap[idx + 1]

        src_tokens = self.src_mmap[start_idx:end_idx]
        tgt_tokens = self.tgt_mmap[start_idx:end_idx]

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)


def binary_collate_fn(batch, pad_idx: int = config.pad_idx):
    """
    Fast collate function to pad batch sequences into (seq_batch, src_lens).
    Compatible with train.py data pipeline expectations.
    """
    src_list, tgt_list = zip(*batch)
    full_seq_list = []
    src_lens = []

    for src, tgt in zip(src_list, tgt_list):
        full_seq = torch.cat([src, tgt], dim=0)
        full_seq_list.append(full_seq)
        # Note: src contains zh_ids + EOS token, so zh_ids length is len(src) - 1
        src_lens.append(len(src) - 1)

    seq_batch = torch.nn.utils.rnn.pad_sequence(full_seq_list, batch_first=True, padding_value=pad_idx)
    src_lens_tensor = torch.tensor(src_lens, dtype=torch.long)
    return seq_batch, src_lens_tensor


def get_binary_dataloader(
    data_dir: str = config.dataset_dir,
    batch_size: int = config.batch_size,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    dataset = BinaryMemmapDataset(data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=binary_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
