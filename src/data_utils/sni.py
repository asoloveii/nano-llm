import os
import argparse
from math import floor

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.tokenizer import Tokenizer


def process_sni(tokenizer: Tokenizer,
                save_dir: str,
                num_examples: int = 1_900_000,
                val_ratio: float = 0.1,
                max_len: int = 1024,
                flush_every: int = 10_000):
    '''
    Tokenizes the Supernatural Instructions dataset (prompt-response pairs) and stores
    them in memory-mapped arrays of shape (num_examples, 2, max_len).

    Args:
        tokenizer (Any): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        num_examples (int, optional): Number of prompt-response pairs to load (defaults to 1_900_000)
        val_ratio (int, optional): How much of the dataset will be used for validation (defaults to 0.1)
        max_len(int, optional): Maximum length of a prompt/response (defaults to 1024).
        flush_every(int, optional): Number of tokens to process before flushing the memmap to disk.
                                     Helps prevent data loss in case of interruption (defaults to 10_000).
    '''
    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "sni_train.npy")
    val_path = os.path.join(save_dir, "sni_val.npy")

    val_examples = floor(num_examples * val_ratio)
    train_examples = num_examples - val_examples

    # create memory mapped array
    dtype = np.uint32 if tokenizer.sp.vocab_size() > 65535 else np.uint16

    train_mmap = np.memmap(train_path, dtype=dtype, mode="w+", shape=(train_examples, 2, max_len))
    val_mmap = np.memmap(val_path, dtype=dtype, mode="w+", shape=(val_examples, 2, max_len))

    # keep track of number of examples loaded
    train_cursor = 0
    val_cursor = 0

    # load dataset with streaming and shuffle data
    dataset_stream = load_dataset(
        "andersonbcdefg/supernatural-instructions-2m", 
        split="train", 
        streaming=True, 
        trust_remote_code=True
    )
    dataset_stream = dataset_stream.shuffle(buffer_size=50_000, seed=42)

    for example in tqdm(dataset_stream, desc="Tokenizing SuperNaturalInstructions"):
        
        prompt = example.get("prompt", "")
        response = example.get("response", "")

        if not prompt.strip() or not response.strip():
            continue

        try:
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response, add_eos=True)
        except Exception:
            print("Encountered tokenizer error, skipping...")
            continue
            
        # pad sequence
        prompt_ids = tokenizer.pad_sequences([prompt_ids], max_len)[0].numpy()
        response_ids = tokenizer.pad_sequences([response_ids], max_len)[0].numpy()

        # decide whether to write to val or train
        if val_cursor < val_examples:
            val_mmap[val_cursor, 0, :max_len] = prompt_ids
            val_mmap[val_cursor, 1, :max_len] = response_ids
            val_cursor += 1
        elif train_cursor < train_examples:
            train_mmap[train_cursor, 0, :max_len] = prompt_ids
            train_mmap[train_cursor, 1, :max_len] = response_ids
            train_cursor += 1
        else:
            break

        if train_cursor % flush_every == 0:
            train_mmap.flush()
        if val_cursor % flush_every == 0:
            val_mmap.flush()
            
    train_mmap.flush()
    val_mmap.flush()
    
    print(f"Saved {train_cursor} examples to {train_path}")
    print(f"Saved {val_cursor} examples to {val_path}")


class InstructionDataset(Dataset):

    def __init__(self, 
                 tokens_path: str, 
                 num_examples: int, 
                 max_seq_len: int, 
                 dtype=np.uint16):
        """
        Dataset for instruction tuning.
        Each example consists of a prompt and a response.

        Args:
            tokens_path (str): Path to the memmap .npy file with shape [num_examples, 2, max_len].
                - data[i, 0, :] = prompt tokens (padded/truncated to max_len)
                - data[i, 1, :] = response tokens (padded/truncated to max_len)
            num_examples (int): Number of examples to load.
            max_seq_len (int): Maximum number of tokens per prompt/response.
            dtype (np.dtype): Data type used in memmap (default: np.uint16).
        """
        super().__init__()
        
        self.data = np.memmap(tokens_path, dtype=dtype, mode="r", 
                              shape=(num_examples, 2, max_seq_len))
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len
        print(self.data.shape)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx: int):
        prompt = torch.tensor(self.data[idx, 0, :self.max_seq_len], dtype=torch.long)
        response = torch.tensor(self.data[idx, 1, :self.max_seq_len], dtype=torch.long)

        # concatenate prompt + response
        x = torch.cat([prompt, response])

        # build labels: -100 for prompt part, actual tokens for response
        labels = torch.cat([
            torch.full_like(prompt, fill_value=-100),  # ignore prompt in loss
            response
        ])

        return x, labels
    

def get_sni_dataloaders(data_dir: str,
                        max_seq_len: int,
                        batch_size: int,
                        num_examples: int = 1_900_000,
                        val_ratio: float = 0.1, 
                        num_workers: int = 0,
                        pin_memory: bool = False,
                        distributed: bool = False):
    """
    Utility function to build a DataLoader for either pretraining or instruction tuning.

    Args:
        data_dir (str): Directory containing memmap .npy files.
        max_seq_len (int): Length of prompt/response.
        batch_size (int): Batch size per step.
        num_examples (int): Number of records to load.
        val_ratio (float): Portion of examples for validation.
        num_workers (int, optional): DataLoader workers (defaults to 0).
        pin_memory (bool, optional): Pin memory for faster host→device transfers.

    Returns:
        (train_loader, val_loader)
    """
    train_path = os.path.join(data_dir, "sni_train.npy")
    val_path   = os.path.join(data_dir, "sni_val.npy")

    n_val = floor(num_examples * val_ratio)
    n_train = num_examples - n_val

    # Load once; shape is inferred automatically
    train_dataset = InstructionDataset(train_path, n_train, max_seq_len)
    val_dataset   = InstructionDataset(val_path,   n_val,   max_seq_len)

    train_sampler = DistributedSampler(train_dataset) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Tokenize and save datasets as memmap arrays.")

    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/processed/sni")
    parser.add_argument("--num_examples", type=int, default=1_900_000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--flush_every", type=int, default=10_000)

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path)

    process_sni(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        num_examples=args.num_examples,
        val_ratio=args.val_ratio,
        max_len=args.max_len,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
