import os
import argparse

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.tokenizer import Tokenizer


def process_openwebtext(tokenizer: Tokenizer, 
                        save_dir: str, 
                        max_tokens_train: int,
                        max_tokens_val: int,
                        flush_every: int = 50_000):
    '''
    Tokenizes the OpenWebText dataset and saves tokens into separate train and validation memory-mapped files.

    Args:
        tokenizer (Any): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        max_tokens_train (int): Maximum number of training tokens to store in the memmap.
        max_tokens_val (int): Maximum number of validation tokens to store in the memmap (defaults to 100_000).
        flush_every(int): Number of tokens to process before flushing the memmap to disk.
                        Helps prevent data loss in case of interruption (defaults to 50_000).
    '''

    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "owt_train.npy")
    val_path = os.path.join(save_dir, "owt_val.npy")

    # create memory mapped array
    dtype = np.uint32 if tokenizer.sp.vocab_size() > 65535 else np.uint16

    train_mmap = np.memmap(train_path, dtype=dtype, mode="w+", shape=(max_tokens_train,))
    val_mmap = np.memmap(val_path, dtype=dtype, mode="w+", shape=(max_tokens_val,))

    # keep track of tokens count
    train_cursor = 0
    val_cursor = 0

    # load dataset with streaming and shuffle data
    dataset_stream = load_dataset(
        "Skylion007/openwebtext", 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )

    dataset_stream = dataset_stream.shuffle(buffer_size=50_000, seed=42)

    for example in tqdm(dataset_stream, desc="Tokenizing OpenWebText"):
        text = example.get("text", "")

        if not text:
            continue # skip empty example

        try:
            ids = tokenizer.encode(text, add_eos=True)
        except Exception:
            print("Encountered tokenizer error, skipping...")
            continue

        if val_cursor < max_tokens_val:
            n = min(len(ids), max_tokens_val - val_cursor)
            val_mmap[val_cursor:val_cursor+n] = ids[:n]
            val_cursor += n
        elif train_cursor < max_tokens_train:
            n = min(len(ids), max_tokens_train - train_cursor)
            train_mmap[train_cursor:train_cursor+n] = ids[:n]
            train_cursor += n
        else:
            break

        if train_cursor % flush_every == 0:
            train_mmap.flush()
        if val_cursor % flush_every == 0:
            val_mmap.flush()

    # final flush
    train_mmap.flush()
    val_mmap.flush()

    print(f"Saved {train_cursor} tokens to {train_path}")
    print(f"Saved {val_cursor} tokens to {val_path}")


class OpenWebTextDataset(Dataset):

    def __init__(self, tokens_path: str, seq_len: int, 
                 dtype=np.uint16, randomize: bool = False):
        """
        Dataset for plain text pretraining (OpenWebText).
        Provides random or sequential slices of tokens for autoregressive training.

        Args:
            tokens_path (str): Path to the memmap .npy file of tokens.
            seq_len (int): Length of token sequences to sample.
            dtype (np.dtype): Data type used in memmap (default: np.uint16).
            randomize (bool): If True, sample random windows; else use sequential windows.
        """
        super().__init__()

        self.data = np.memmap(tokens_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.total_tokens = len(self.data)
        self.randomize = randomize

        if self.total_tokens <= seq_len + 2:
            raise RuntimeError(
                f"Not enough tokens ({self.total_tokens}) for seq_len={seq_len}."
            )

    def __len__(self):
        # return the number of non-overlapping sequences
        return (self.total_tokens - 1) // self.seq_len

    def __getitem__(self, idx: int):
        if self.randomize:
            # randomly choosing a chunk from data usually helps with generalization
            start = torch.randint(0, self.total_tokens - self.seq_len - 1, (1,)).item()
        else:
            start = idx * self.seq_len

        x = torch.tensor(self.data[start:start+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start+1:start+self.seq_len+1], dtype=torch.long)

        return x, y
    

def get_owt_dataloaders(data_dir: str,
                        max_seq_len: int,
                        batch_size: int,
                        num_workers: int = 0,
                        pin_memory: bool = False,
                        distributed: bool = False,
                        random_sample: bool = False):
    """
    Utility function to build a DataLoader for either pretraining or instruction tuning.

    Args:
        data_dir (str): Directory containing memmap .npy files.
        max_seq_len (int): Sequence length.
        batch_size (int): Batch size per step.
        num_workers (int, optional): DataLoader workers (defaults to 0).
        pin_memory (bool, optional): Pin memory for faster host to device transfers.
        distributed (bool, optional): Data distributed parallelism.
        randome_sample (boo, optional): Sample a random sequence from processed data.

    Returns:
        (train_loader, val_loader)
    """
    train_path = os.path.join(data_dir, f"owt_train.npy")
    val_path = os.path.join(data_dir, f"owt_val.npy")

    train_dataset = OpenWebTextDataset(train_path, max_seq_len, randomize=random_sample)
    val_dataset = OpenWebTextDataset(val_path, max_seq_len, randomize=False)
    
    train_dataset = OpenWebTextDataset(
        train_path, seq_len=max_seq_len, randomize=random_sample
    )
    val_dataset = OpenWebTextDataset(
        val_path, seq_len=max_seq_len, randomize=False
    )

    train_sampler = DistributedSampler(train_dataset) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),   # no shuffle if using distributed sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Tokenize and save OWT dataset as memmap arrays.")

    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/processed/owt")
    parser.add_argument("--max_tokens_train", type=int, required=True)
    parser.add_argument("--max_tokens_val", type=int, required=True)
    parser.add_argument("--flush_every", type=int, default=50_000)

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path)

    process_openwebtext(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_tokens_train=args.max_tokens_train,
        max_tokens_val=args.max_tokens_val,
        flush_every=args.flush_every
    )


if __name__ == "__main__":
    main()