import gzip
import json
import glob
import os
import random
import re
from typing import List, Tuple, Generator, Iterator
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from config import config
from vocab_tokenizer import VocabTokenizer

class ParallelTextDataset(Dataset):
    """
    In-memory dataset reader for a specific number of samples.
    """
    def __init__(
        self,
        data_dir: str = config.dataset_dir,
        tokenizer: VocabTokenizer = None,
        max_samples: int = 10000,
        max_seq_len: int = config.max_seq_len
    ):
        self.tokenizer = tokenizer or VocabTokenizer()
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []

        # Hỗ trợ nạp kết hợp nhiều thư mục/tệp dataset (ngăn cách bằng dấu phẩy hoặc dấu cộng)
        if isinstance(data_dir, str):
            dirs = [d.strip() for d in re.split(r'[,+]', data_dir) if d.strip()]
        elif isinstance(data_dir, (list, tuple)):
            dirs = list(data_dir)
        else:
            dirs = [data_dir]

        files = []
        for d in dirs:
            if os.path.isfile(d):
                files.append(d)
            elif os.path.isdir(d):
                files.extend(sorted(glob.glob(os.path.join(d, "*.jsonl.gz"))))
                files.extend(sorted(glob.glob(os.path.join(d, "*.jsonl"))))

        if not files:
            raise FileNotFoundError(f"No .jsonl or .jsonl.gz files found in: {data_dir}")

        count = 0
        for file_path in files:
            if max_samples is not None and count >= max_samples:
                break
            try:
                open_fn = gzip.open(file_path, "rt", encoding="utf-8") if file_path.endswith(".gz") else open(file_path, "r", encoding="utf-8")
                with open_fn as f:
                    for line in f:
                        if max_samples is not None and count >= max_samples:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        zh_text = data.get("prompt_src") or data.get("zh_text", "").strip()
                        vi_text = data.get("vi_text", "").strip()

                        if not zh_text or not vi_text:
                            continue

                        zh_ids = self.tokenizer.encode_zh(zh_text, add_bos_eos=False)
                        vi_ids = self.tokenizer.encode_vi(vi_text, add_bos_eos=True)

                        if len(zh_ids) > 1000 or len(vi_ids) > 1000 or (len(zh_ids) + 1 + len(vi_ids)) > self.max_seq_len:
                            continue

                        self.samples.append((zh_ids, vi_ids))
                        count += 1
            except Exception as e:
                print(f"Warning reading {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]

class StreamingParallelDataset(IterableDataset):
    """
    High-performance streaming dataset that iterates through ALL 235 zipped JSONL files.
    """
    def __init__(
        self,
        data_dir: str = config.dataset_dir,
        tokenizer: VocabTokenizer = None,
        max_seq_len: int = config.max_seq_len,
        shuffle_files: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer or VocabTokenizer()
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        if isinstance(self.data_dir, str):
            dirs = [d.strip() for d in re.split(r'[,+]', self.data_dir) if d.strip()]
        elif isinstance(self.data_dir, (list, tuple)):
            dirs = list(self.data_dir)
        else:
            dirs = [self.data_dir]

        gz_files = []
        for d in dirs:
            if os.path.isfile(d):
                gz_files.append(d)
            elif os.path.isdir(d):
                gz_files.extend(sorted(glob.glob(os.path.join(d, "*.jsonl.gz"))))
                gz_files.extend(sorted(glob.glob(os.path.join(d, "*.jsonl"))))

        if not gz_files:
            raise FileNotFoundError(f"No .jsonl or .jsonl.gz files found in directory: {self.data_dir}")

        if self.shuffle_files:
            random.shuffle(gz_files)

        for file_path in gz_files:
            try:
                open_fn = gzip.open(file_path, "rt", encoding="utf-8") if file_path.endswith(".gz") else open(file_path, "r", encoding="utf-8")
                with open_fn as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        zh_text = data.get("prompt_src") or data.get("zh_text", "").strip()
                        vi_text = data.get("vi_text", "").strip()

                        if not zh_text or not vi_text:
                            continue

                        zh_ids = self.tokenizer.encode_zh(zh_text, add_bos_eos=False)
                        vi_ids = self.tokenizer.encode_vi(vi_text, add_bos_eos=True)

                        # Safe sequence length limit to prevent unexpected 2048-token OOM spikes during backward pass
                        if len(zh_ids) > 256 or len(vi_ids) > 256 or (len(zh_ids) + 1 + len(vi_ids)) > 512:
                            continue

                        yield (zh_ids, vi_ids)
            except Exception as e:
                print(f"Warning reading {file_path}: {e}")

class CMLMSeqCollate:
    """
    Unpadded Concatenation & Right-End Padding Collate Function.
    Prevents Middle-Padding Pollution in SSM Recurrence.
    Pushes ALL PAD tokens 100% to the right end of the sequence.
    """
    def __init__(self, pad_idx: int = config.pad_idx, eos_idx: int = config.eos_idx):
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __call__(self, batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_seqs = []
        src_lens = []

        for zh_ids, vi_ids in batch:
            # Format: [ZH_ids..., EOS_IDX, VI_ids (which starts with BOS_IDX and ends with EOS_IDX)]
            seq = zh_ids + [self.eos_idx] + vi_ids
            combined_seqs.append(seq)
            src_lens.append(len(zh_ids))

        max_len = max(len(s) for s in combined_seqs)
        padded_seqs = [s + [self.pad_idx] * (max_len - len(s)) for s in combined_seqs]

        seq_tensor = torch.tensor(padded_seqs, dtype=torch.long)
        src_len_tensor = torch.tensor(src_lens, dtype=torch.long)

        return seq_tensor, src_len_tensor

# Alias for backward compatibility
Seq2SeqCollate = CMLMSeqCollate

def get_dataloader(
    data_dir: str = config.dataset_dir,
    batch_size: int = config.batch_size,
    max_samples: int = None,
    streaming: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, VocabTokenizer]:
    tokenizer = VocabTokenizer()
    collate_fn = CMLMSeqCollate(pad_idx=config.pad_idx, eos_idx=config.eos_idx)

    use_pin_memory = pin_memory and torch.cuda.is_available()

    if streaming or max_samples is None:
        dataset = StreamingParallelDataset(data_dir=data_dir, tokenizer=tokenizer, shuffle_files=shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=use_pin_memory,
        )
    else:
        dataset = ParallelTextDataset(data_dir=data_dir, tokenizer=tokenizer, max_samples=max_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

    return dataloader, tokenizer

if __name__ == "__main__":
    print("Testing CMLM Right-Padded Streaming DataLoader...")
    loader, tok = get_dataloader(streaming=True, batch_size=4)
    for i, (seq_batch, src_lens) in enumerate(loader):
        print(f"Batch {i+1} - seq shape: {seq_batch.shape}, src_lens: {src_lens.tolist()}")
        if i >= 2:
            break
