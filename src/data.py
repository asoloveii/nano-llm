import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class OpenWebTextDataset(Dataset):

    def __init__(self, tokens_path: str, seq_len: int, dtype=np.uint32, randomize: bool = True):
        """
        Dataset for plain text pretraining (e.g., OpenWebText).
        Provides random or sequential slices of tokens for autoregressive training.

        Args:
            tokens_path (str): Path to the memmap .dat file of tokens.
            seq_len (int): Length of token sequences to sample.
            dtype (np.dtype): Data type used in memmap (default: np.uint32).
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
        return self.total_tokens // self.seq_len

    def __getitem__(self, idx: int):
        if self.randomize:
            start = torch.randint(0, self.total_tokens - self.seq_len - 1, (1,)).item()
        else:
            start = idx * self.seq_len

        x = self.data[start:start+self.seq_len]
        y = self.data[start+1:start+self.seq_len+1]

        return (torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(y.astype(np.int64)))
    

class InstructionDataset(Dataset):

    def __init__(self, tokens_path: str, max_prompt_len: int = 256, max_response_len: int = 256):
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
        self.data = np.memmap(tokens_path, dtype=np.uint32, mode="r")
        self.num_examples = self.data.shape[0]
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.total_len = max_prompt_len + max_response_len

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        prompt = torch.from_numpy(self.data[idx, 0, :self.max_prompt_len].astype(np.int64))
        response = torch.from_numpy(self.data[idx, 1, :self.max_response_len].astype(np.int64))

        # concatenate prompt + response
        x = torch.cat([prompt, response])

        # build labels: -100 for prompt part, actual tokens for response
        labels = torch.cat([
            torch.full_like(prompt, fill_value=-100),  # ignore prompt in loss
            response
        ])

        return x, labels


def get_dataloader(dataset_type: str, 
                   dataset_name: str,
                   data_path: str, 
                   seq_len: int, 
                   batch_size: int, 
                   num_workers: int = 2):
    """
    Utility function to build a DataLoader for either pretraining or instruction tuning.

    Args:
        dataset_type (str): "pretraining" or "instruction".
        tokens_path (str): Path to memmap .dat file.
        seq_len (int): Sequence length for pretraining dataset (ignored for instruction dataset).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of DataLoader workers.

    Returns:
        DataLoader: Iterable over batches of (x, y).
    """
    # define path
    train_path = os.path.join(data_path, f"{dataset_name}_train.dat")
    val_path = os.path.join(data_path, f"{dataset_name}_validation.dat")

    # use appropriate dataset class
    if dataset_type == "pretraining":
        train_dataset = OpenWebTextDataset(train_path, seq_len)
        val_dataset = OpenWebTextDataset(val_path, seq_len)
    elif dataset_type == "instruction":
        train_dataset = InstructionDataset(train_path, max_prompt_len=seq_len//2, 
                                           max_response_len=seq_len//2)
        val_dataset = InstructionDataset(val_path, max_prompt_len=seq_len//2, 
                                         max_response_len=seq_len//2)
    
    # return data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
