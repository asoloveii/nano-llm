import os
from math import floor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class OpenWebTextDataset(Dataset):

    def __init__(self, tokens_path: str, seq_len: int, 
                 dtype=np.uint16, randomize: bool = True):
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

    def __len__(self):
        # return the number of non-overlapping sequences
        return max(1, (self.total_tokens - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        if self.total_tokens <= self.seq_len + 1:
            raise ValueError("Sequence length is too long for available tokens.")
        
        if self.randomize:
            # randomly choosing a chunk from data usually helps with generalization
            start = torch.randint(0, self.total_tokens - self.seq_len - 1, (1,)).item()
        else:
            start = idx * self.seq_len

        x = torch.tensor(self.data[start:start+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[start+1:start+self.seq_len+1], dtype=torch.long)

        return x, y
    

class InstructionDataset(Dataset):

    def __init__(self, tokens_path: str, num_examples: int, 
                 max_len: int = 1024, dtype=np.uint16):
        """
        Dataset for instruction tuning.
        Each example consists of a prompt and a response.

        Args:
            tokens_path (str): Path to the memmap .npy file with shape [num_examples, 2, max_len].
                - data[i, 0, :] = prompt tokens (padded/truncated to max_len)
                - data[i, 1, :] = response tokens (padded/truncated to max_len)
            num_examples (int): Number of examples to load.
            max_len (int): Maximum number of tokens per prompt/response.
            dtype (np.dtype): Data type used in memmap (default: np.uint16).
        """
        self.data = np.memmap(tokens_path, dtype=dtype, mode="r", 
                              shape=(num_examples, 2, max_len))
        self.num_examples = num_examples
        self.max_len = max_len
        print(self.data.shape)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx: int):
        prompt = torch.tensor(self.data[idx, 0, :self.max_len], dtype=torch.long)
        response = torch.tensor(self.data[idx, 1, :self.max_len], dtype=torch.long)

        # concatenate prompt + response
        x = torch.cat([prompt, response])

        # build labels: -100 for prompt part, actual tokens for response
        labels = torch.cat([
            torch.full_like(prompt, fill_value=-100),  # ignore prompt in loss
            response
        ])

        return x, labels
    

class GSM8KDataset(Dataset):

    def __init__(self, q_path: str, ans_path: str, num_examples: int, 
                 max_seq_len: int, q_dtype=np.uint16, ans_dtype=np.int32):
        '''
        Dataset for GSM8K math reasoning tasks.
        Each item is a (question, answer) pair.

        Args:
            q_path (str): Path to question memmap, shape [num_examples, max_seq_len].
            ans_path (str): Path to answer memmap, shape [num_examples].
            num_examples (int): Number of examples to load.
            max_seq_len (int): Maximum tokens per question.
            q_dtype (np.dtype, optional): Question dtype (defaults to np.uint16).
            ans_dtype (np.dtype, optional): Answer dtype (defaults to np.int32)
        '''
        self.questions = np.memmap(q_path, dtype=q_dtype, mode="r", 
                                   shape=(num_examples, max_seq_len))
        self.answers = np.memmap(ans_path, dtype=ans_dtype, mode="r", 
                                 shape=(num_examples, ))
        self.num_examples = num_examples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx: int):
        q = torch.tensor(self.questions[idx, :], dtype=torch.long)
        ans = torch.tensor(self.answers[idx], dtype=torch.long)

        return q, ans


def get_dataloaders(dataset: str,
                    data_dir: str,
                    seq_len: int,
                    batch_size: int,
                    num_examples: int = 1_900_000,
                    val_ratio: float = 0.1, 
                    num_workers: int = 0,
                    pin_memory: bool = False,
                    distributed: bool = False):
    """
    Utility function to build a DataLoader for either pretraining or instruction tuning.

    Args:
        dataset (str): One of {"owt", "sni"}.
        data_dir (str): Directory containing memmap .npy files.
        seq_len (int): Sequence length (for OWT) or length of prompt/response (for SNI).
        batch_size (int): Batch size per step.
        num_examples (int): Number of examples (for SNI).
        val_ratio (float): Portion of examples for validation (for SNI).
        num_workers (int, optional): DataLoader workers (defaults to 0).
        pin_memory (bool, optional): Pin memory for faster host→device transfers.

    Returns:
        (train_loader, val_loader)
    """
    train_path = os.path.join(data_dir, f"{dataset}_train.npy")
    val_path = os.path.join(data_dir, f"{dataset}_val.npy")

    # use appropriate dataset class
    if dataset == "owt":
        train_dataset = OpenWebTextDataset(train_path, seq_len)
        val_dataset = OpenWebTextDataset(val_path, seq_len, randomize=False)
    elif dataset == "sni":
        half_len = seq_len // 2
        n_val = floor(num_examples * val_ratio)
        n_train = num_examples - n_val
        train_dataset = InstructionDataset(train_path, n_train, max_len=half_len)
        val_dataset = InstructionDataset(val_path, n_val, max_len=half_len)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory, 
                                persistent_workers=num_workers > 0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=num_workers > 0)
    
    return train_loader, val_loader


def get_gsm8k_dataloader(data_path: str, 
                         split: str,
                         num_examples: int,
                         max_seq_len: int,
                         batch_size: int,
                         num_workers: int = 0,
                         pin_memory: bool = False,
                         distributed: bool = False):
    '''
    Build DataLoader for GSM8K math reasoning tasks.

    Args:
        data_dir (str): Directory with GSM8K memmap files.
        split (str): Dataset split ("train" or "test").
        num_examples (int): Number of examples to load.
        max_seq_len (int): Max sequence length per question.
        batch_size (int): Batch size.
        num_workers (int, optional): DataLoader workers (defaults to 0).
        pin_memory (bool, optional): Pin memory for faster host→device transfers.

    Returns:
        DataLoader: Iterable over batches of (question, answer).
    '''
    q_path = os.path.join(data_path, f"gsm8k_q_{split}.npy")
    ans_path = os.path.join(data_path, f"gsm8k_ans_{split}.npy")

    dataset = GSM8KDataset(q_path, ans_path, num_examples, max_seq_len)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    dataloader = DataLoader(dataset, sampler=sampler, shuffle=(split == "train"), 
                                batch_size=batch_size, num_workers=num_workers, 
                                pin_memory=pin_memory,
                                persistent_workers=num_workers > 0)

    return dataloader
