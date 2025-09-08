import os
import argparse
from math import floor

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import Tokenizer


def process_openwebtext(tokenizer: Tokenizer, 
                        save_dir: str, 
                        max_tokens_train: int = 1_900_000_000,
                        max_tokens_val: int = 100_000) -> None:
    '''
    Tokenizes the OpenWebText dataset and saves tokens into separate train and validation memory-mapped files.

    Args:
        tokenizer (Any): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        max_tokens_train (int, optional): Maximum number of training tokens 
                                          to store in the memmap (defaults to 1_900_000_000).
        max_tokens_val (int, optional): Maximum number of validation tokens 
                                          to store in the memmap (defaults to 100_000).
    '''

    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "owt_train.npy")
    val_path = os.path.join(save_dir, "owt_val.npy")

    # create memory mapped array
    train_mmap = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(max_tokens_train,))
    val_mmap = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(max_tokens_val,))
    # keep track of tokens count
    train_cursor = 0
    val_cursor = 0

    # load dataset with streaming and shuffle data
    dataset_stream = load_dataset("Skylion007/openwebtext", split="train", 
                                  streaming=True, trust_remote_code=True)
    dataset_stream = dataset_stream.shuffle(buffer_size=10_000, seed=42)

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

    train_mmap.flush()
    val_mmap.flush()
    print(f"Saved {train_cursor} tokens to {train_path}")
    print(f"Saved {val_cursor} tokens to {val_path}")


def process_sni(tokenizer: Tokenizer,
                save_dir: str,
                num_examples: int = 1_900_000,
                val_ratio: float = 0.1,
                max_len: int = 1024):
    '''
    
    '''
    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "sni_train.npy")
    val_path = os.path.join(save_dir, "sni_val.npy")

    val_examples = floor(num_examples * val_ratio)
    train_examples = num_examples - val_examples

    # create memory mapped array
    train_mmap = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(train_examples, 2, max_len))
    val_mmap = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(val_examples, 2, max_len))
    # keep track of number of examples loaded
    train_cursor = 0
    val_cursor = 0

    # load dataset with streaming and shuffle data
    dataset_stream = load_dataset("andersonbcdefg/supernatural-instructions-2m", split="train", 
                                  streaming=True, trust_remote_code=True)
    dataset_stream = dataset_stream.shuffle(buffer_size=10_000, seed=42)

    for example in dataset_stream:
        prompt = example["prompt"]
        response = example["response"]

        try:
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response, add_eos=True)
        except Exception:
            print("Encountered tokenizer error, skipping...")
            continue
            
        # pad sequence
        prompt_pad = [tokenizer.pad_id] * (max_len - len(prompt_ids))
        response_pad = [tokenizer.pad_id] * (max_len - len(response_ids))

        prompt_ids += prompt_pad 
        response_ids += response_pad

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
            
    train_mmap.flush()
    val_mmap.flush()
    print(f"Saved {train_cursor} examples to {train_path}")
    print(f"Saved {val_cursor} examples to {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize and save datasets as memmap files.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["openwebtext", "sni", "gsm8k"])
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the trained sentencepiece tokenizer .model")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Directory in which tokenized datasets will be stored")
    parser.add_argument("--num_examples", type=int, default=25, 
                        help="Only for instruction datasets, total number of examples to load")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Ratio of validation set from num_examples")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Only for instruction datasets, maximum length of a prompt/response")
    parser.add_argument("--max_tokens_train", type=int, default=1_000,
                        help="Only for openwebtext, maximum total training tokens to store")
    parser.add_argument("--max_tokens_val", type=int, default=500,
                        help="Only for openwebtext, maximum total validation tokens to store")
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path)

    if args.dataset == "openwebtext":
        process_openwebtext(tokenizer, args.save_dir, args.max_tokens_train, args.max_tokens_val)
    elif args.dataset == "sni":
        process_sni(tokenizer, args.save_dir, args.num_examples, args.val_ratio, args.max_len)


if __name__ == "__main__":
    main()
