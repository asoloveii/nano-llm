import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class OpenWebTextDataset(Dataset):

    def __init__(self, tokens_path: str, seq_len: int, dtype=np.uint16, randomize: bool = True):
        """
        Dataset for plain text pretraining (OpenWebText).
        Provides random or sequential slices of tokens for autoregressive training.

        Args:
            tokens_path (str): Path to the memmap .npy file of tokens.
            seq_len (int): Length of token sequences to sample.
            dtype (np.dtype): Data type used in memmap (default: np.uint16).
            randomize (bool): If True, sample random windows; else use sequential windows.

        Returns:
            x (torch.LongTensor): Input token IDs of shape [seq_len].
            y (torch.LongTensor): Target token IDs (next tokens), same shape [seq_len].
        """
        super().__init__()
        self.data = np.memmap(tokens_path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.total_tokens = len(self.data)
        self.randomize = randomize

    def __len__(self):
        # return the number of distince sequences
        return (self.total_tokens - 1) // self.seq_len

    def __getitem__(self, idx: int):
        if self.randomize:
            # randomly choosing a chunk from data usually helps with generalization
            start = torch.randint(0, self.total_tokens - self.seq_len - 1, (1,)).item()
        else:
            start = idx * self.seq_len

        x = torch.as_tensor(self.data[start:start+self.seq_len], dtype=torch.long)
        y = torch.as_tensor(self.data[start+1:start+self.seq_len+1], dtype=torch.long)

        return x, y
    

class InstructionDataset(Dataset):

    def __init__(self, tokens_path: str, num_examples: int, max_len: int = 1024):
        """
        Dataset for instruction tuning.
        Each example consists of a prompt and a response.

        Args:
            tokens_path (str): Path to the memmap .dat file with shape [num_examples, 2, max_len].
                - data[i, 0, :] = prompt tokens (padded/truncated to max_prompt_len)
                - data[i, 1, :] = response tokens (padded/truncated to max_response_len)
            max_prompt_len (int): Maximum number of prompt tokens.
            max_response_len (int): Maximum number of response tokens.

        Returns:
            x (torch.LongTensor): Input tokens (prompt + response), shape [max_prompt_len + max_response_len].
            labels (torch.LongTensor): Labels aligned with x, where:
                - prompt positions = -100 (ignored in loss)
                - response positions = token IDs (used for loss)
        """
        self.data = np.memmap(tokens_path, dtype=np.uint16, mode="r", shape=(num_examples, 2, max_len))
        self.num_examples = num_examples
        self.max_len = max_len

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        print(self.data.shape)
        prompt = torch.as_tensor(self.data[idx, 0, :self.max_len], dtype=torch.long)
        response = torch.as_tensor(self.data[idx, 1, :self.max_len], dtype=torch.long)

        # concatenate prompt + response
        x = torch.cat([prompt, response])

        # build labels: -100 for prompt part, actual tokens for response
        labels = torch.cat([
            torch.full_like(prompt, fill_value=-100),  # ignore prompt in loss
            response
        ])

        return x, labels


def get_dataloader(dataset: str,
                   data_dir: str, 
                   num_examples: int,
                   seq_len: int, 
                   batch_size: int, 
                   num_workers: int = 0):
    """
    Utility function to build a DataLoader for either pretraining or instruction tuning.

    Args:
        dataset (str): "owt", "sni" or "gsm8k".
        tokens_path (str): Path to memmap .dat file.
        seq_len (int): Sequence length for pretraining dataset (ignored for instruction dataset).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of DataLoader workers.

    Returns:
        DataLoader: Iterable over batches of (x, y).
    """
    train_path = os.path.join(data_dir, f"{dataset}_train.npy")
    val_path = os.path.join(data_dir, f"{dataset}_val.npy")

    # use appropriate dataset class
    if dataset == "owt":
        train_dataset = OpenWebTextDataset(train_path, seq_len)
        val_dataset = OpenWebTextDataset(val_path, seq_len)
    elif dataset == "sni":
        train_dataset = InstructionDataset(train_path, num_examples, max_len=seq_len//2)
        val_dataset = InstructionDataset(val_path, num_examples, max_len=seq_len//2)
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader
