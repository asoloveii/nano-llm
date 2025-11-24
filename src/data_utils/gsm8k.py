import os
import argparse

import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.tokenizer import Tokenizer


def process_gsm8k(tokenizer: Tokenizer,
                  save_dir: str,
                  max_seq_len: int,
                  split: str = "train",
                  flush_every: int = 1_000,
                  shuffle: bool = True):
    '''
    Tokenizes GSM8K math dataset (questions and numeric answers) and stores in memmap arrays.

    Args:
        tokenizer (Tokenizer): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        max_seq_len (int, optional): Maximum length of a question.
        split (str, optional): train/test split of gsm8k dataset (defaults to train).
        flush_every (int, optional): Number of tokens to process before flushing the memmap to disk.
                                     Helps prevent data loss in case of interruption (defaults to 1_000).
        shuffle (bool, optional): Randomly shuffle dataset
    '''
    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    q_path = os.path.join(save_dir, f"gsm8k_q_{split}.npy")
    ans_path = os.path.join(save_dir, f"gsm8k_ans_{split}.npy")

    # create memmap arrays for question and a final answer(for GRPO)
    split = "test" if split == "val" else split
    dataset = load_dataset("openai/gsm8k", "main", split=split, trust_remote_code=True)
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    num_examples = len(dataset)
    dtype = np.uint32 if tokenizer.sp.vocab_size() > 65535 else np.uint16
    questions_mmap = np.memmap(q_path, dtype=dtype, mode="w+", shape=(num_examples, max_seq_len))
    answers_mmap = np.memmap(ans_path, dtype=np.float32, mode="w+", shape=(num_examples, ))

    cursor = 0

    for example in tqdm(dataset, desc="Tokenizing math questions"):

        q = example["question"]
        answer = example["answer"]
        
        if "####" in answer:
            answer_str = answer.split("####")[-1].strip()
            ans = float(answer_str)
        else:
            continue  # skip malformed record

        try:
            q_ids = tokenizer.encode(q, add_bos=True, add_eos=True)
        except Exception:
            print("Encountered an error, skipping...")
            continue
        
        # pad question sequence
        q_ids = tokenizer.pad_sequences([q_ids], max_seq_len)[0].numpy()

        questions_mmap[cursor, :max_seq_len] = q_ids 
        answers_mmap[cursor] = float(ans)   # save just a numeric value

        cursor += 1

        if cursor % flush_every == 0:
            questions_mmap.flush()
            answers_mmap.flush()
    
    questions_mmap.flush()
    answers_mmap.flush()
    print(f"Saved {cursor} questions/answers into {q_path}, {ans_path}")


class GSM8KDataset(Dataset):

    def __init__(self, 
                 q_path: str, 
                 ans_path: str, 
                 max_seq_len: int, 
                 q_dtype=np.uint16, 
                 ans_dtype=np.float32):
        '''
        Dataset for GSM8K math reasoning tasks.
        Each item is a (question, answer) pair.

        Args:
            q_path (str): Path to question memmap, shape [num_examples, max_seq_len].
            ans_path (str): Path to answer memmap, shape [num_examples].
            max_seq_len (int): Maximum tokens per question.
            q_dtype (np.dtype, optional): Question dtype (defaults to np.uint16).
            ans_dtype (np.dtype, optional): Answer dtype (defaults to np.float32)
        '''
        super().__init__()
        
        self.questions = np.memmap(q_path, dtype=q_dtype, mode="r")
        self.answers = np.memmap(ans_path, dtype=ans_dtype, mode="r")

        # infer number of samples
        num_examples = self.answers.shape[0]

        # reshape questions
        self.questions = self.questions.reshape(num_examples, max_seq_len)

        self.num_examples = num_examples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx: int):
        q = torch.tensor(self.questions[idx], dtype=torch.long)
        ans = torch.tensor(self.answers[idx], dtype=torch.float32)

        return q, ans


def get_gsm8k_dataloaders(data_path: str,
                          max_seq_len: int,
                          batch_size: int,
                          num_workers: int = 0,
                          pin_memory: bool = False,):
    '''
    Build DataLoader for GSM8K math reasoning tasks.

    Args:
        data_dir (str): Directory with GSM8K memmap files.
        max_seq_len (int): Max sequence length per question.
        batch_size (int): Batch size.
        num_workers (int, optional): DataLoader workers (defaults to 0).
        pin_memory (bool, optional): Pin memory for faster host→device transfers.

    Returns:
        (train_loader, val_loader)
    '''
    train_q_path = os.path.join(data_path, f"gsm8k_q_train.npy")
    train_ans_path = os.path.join(data_path, f"gsm8k_ans_train.npy")

    val_q_path = os.path.join(data_path, f"gsm8k_q_val.npy")
    val_ans_path = os.path.join(data_path, f"gsm8k_ans_val.npy")

    train_dataset = GSM8KDataset(
        train_q_path, train_ans_path, max_seq_len=max_seq_len
    )

    val_dataset = GSM8KDataset(
        val_q_path, val_ans_path, max_seq_len=max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    parser = argparse.ArgumentParser(description="Tokenize and save GSM8K dataset as memmap arrays.")

    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="data/processed/gsm8k")
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--flush_every", type=int, default=1_000)
    parser.add_argument("--shuffle", type=bool, default=True)

    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path)

    process_gsm8k(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.max_seq_len,
        split=args.split,
        flush_every=args.flush_every,
        shuffle=args.shuffle
    )


if __name__ == "__main__":
    main()