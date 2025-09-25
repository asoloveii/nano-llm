import os
import argparse
from math import floor

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer import Tokenizer


def pad_or_truncate(tokens, max_len, pad_id):
    """Truncate to max_len and pad if shorter."""
    return (tokens[:max_len] + [pad_id] * max_len)[:max_len]


def process_openwebtext(tokenizer: Tokenizer, 
                        save_dir: str, 
                        max_tokens_train: int = 1_900_000_000,
                        max_tokens_val: int = 100_000,
                        flush_every: int = 50_000):
    '''
    Tokenizes the OpenWebText dataset and saves tokens into separate train and validation memory-mapped files.

    Args:
        tokenizer (Any): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        max_tokens_train (int, optional): Maximum number of training tokens 
                                          to store in the memmap (defaults to 1_900_000_000).
        max_tokens_val (int, optional): Maximum number of validation tokens 
                                          to store in the memmap (defaults to 100_000).
        flush_every(int, optional): Number of tokens to process before flushing the memmap to disk.
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

        if train_cursor % flush_every == 0:
            train_mmap.flush()
        if val_cursor % flush_every == 0:
            val_mmap.flush()

    train_mmap.flush()
    val_mmap.flush()
    print(f"Saved {train_cursor} tokens to {train_path}")
    print(f"Saved {val_cursor} tokens to {val_path}")


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
    dataset_stream = load_dataset("andersonbcdefg/supernatural-instructions-2m", split="train", 
                                  streaming=True, trust_remote_code=True)
    dataset_stream = dataset_stream.shuffle(buffer_size=10_000, seed=42)

    for example in tqdm(dataset_stream, desc="Tokenizing instruction"):
        prompt = example["prompt"]
        response = example["response"]

        try:
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response, add_eos=True)
        except Exception:
            print("Encountered tokenizer error, skipping...")
            continue
            
        # pad sequence
        prompt_ids = pad_or_truncate(prompt_ids, max_len, tokenizer.pad_id)
        response_ids = pad_or_truncate(response_ids, max_len, tokenizer.pad_id)

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


def process_gsm8k(tokenizer: Tokenizer,
                  save_dir: str,
                  num_examples: int,
                  max_seq_len: int,
                  split: str = "train",
                  flush_every: int = 1_000):
    '''
    Tokenizes GSM8K math dataset (questions and numeric answers) and stores in memmap arrays.

    Args:
        tokenizer (Any): A sentencepiece tokenizer with an `encode` method.
        save_dir (str): Directory to save the token files.
        num_examples (int, optional): Number of prompt-response pairs to load
        max_seq_len(int, optional): Maximum length of a question.
        split(str, optional): train/test split of gsm8k dataset (defaults to train).
        flush_every(int, optional): Number of tokens to process before flushing the memmap to disk.
                                     Helps prevent data loss in case of interruption (defaults to 1_000).
    '''
    # create path to store memory mapped array
    os.makedirs(save_dir, exist_ok=True)
    q_path = os.path.join(save_dir, f"gsm8k_q_{split}.npy")
    ans_path = os.path.join(save_dir, f"gsm8k_ans_{split}.npy")

    # create memmap arrays for question and a final answer(for GRPO)
    dtype = np.uint32 if tokenizer.sp.vocab_size() > 65535 else np.uint16
    questions_mmap = np.memmap(q_path, dtype=dtype, mode="w+", shape=(num_examples, max_seq_len))
    answers_mmap = np.memmap(ans_path, dtype=np.int32, mode="w+", shape=(num_examples, ))

    # load dataset with streaming and shuffle data
    dataset_stream = load_dataset("openai/gsm8k", "main", split=split, 
                                  streaming=True, trust_remote_code=True)
    dataset_stream = dataset_stream.shuffle(buffer_size=10_000, seed=42)

    cursor = 0 # keep track of number of loaded examples

    for example in tqdm(dataset_stream, desc="Tokenizing math questions"):
        # break a loop if loaded enough examples
        if cursor >= num_examples:
            break

        q = example["question"]
        ans = example["answer"].split("####")[-1].strip()   # only final answer

        try:
            q_ids = tokenizer.encode(q, add_bos=True, add_eos=True)
        except Exception:
            print("Encountered an error, skipping...")
            continue
        
        # pad question sequence
        q_ids = pad_or_truncate(q_ids, max_seq_len, tokenizer.pad_id)

        questions_mmap[cursor, :max_seq_len] = q_ids 
        answers_mmap[cursor] = int(ans)   # save just a numeric value

        cursor += 1

        if cursor % flush_every == 0:
            questions_mmap.flush()
            answers_mmap.flush()
    
    questions_mmap.flush()
    answers_mmap.flush()
    print(f"Saved {cursor} questions/answers into {q_path}, {ans_path}")


def main():
    parser = argparse.ArgumentParser( 
        description="Tokenize and save datasets as memory-mapped arrays for training."
    )
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["openwebtext", "sni", "gsm8k"], help="Which dataset to process.")
    parser.add_argument("--tokenizer_path", type=str, required=True, 
                        help="Path to the trained SentencePiece tokenizer .model file.")
    parser.add_argument("--save_dir", type=str, default="data", 
                        help="Directory to store tokenized datasets. Defaults to './data'.")
    parser.add_argument("--num_examples", type=int, default=1_900_000, 
                        help="For SNI or GSM8K: number of examples to process. Default is 1.9M for SNI.")
    parser.add_argument("--split", type=str, default="train", 
                        help="Dataset split for GSM8K. Default is 'train'.")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="Ratio of validation examples for SNI. Default is 0.1 (10%).")
    parser.add_argument("--max_len", type=int, default=1024, 
                        help="Maximum sequence length for SNI prompts/responses. Default is 1024.")
    parser.add_argument("--max_tokens_train", type=int, default=2_180_000_000,
                        help="Max tokens for OpenWebText training memmap. Default is 2.18B.")
    parser.add_argument("--max_tokens_val", type=int, default=100_000,
                        help="Max tokens for OpenWebText validation memmap. Default is 100k.")
    parser.add_argument("--flush_every", type=int, default=10_000, 
                        help="Number of tokens to process before flushing the memmap to disk. Default is 10k.")
    
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_path)

    if args.dataset == "openwebtext":
        process_openwebtext(tokenizer, args.save_dir, args.max_tokens_train, args.max_tokens_val, args.flush_every)
    elif args.dataset == "sni":
        process_sni(tokenizer, args.save_dir, args.num_examples, args.val_ratio, args.max_len, args.flush_every)
    elif args.dataset == "gsm8k":
        process_gsm8k(tokenizer, args.save_dir, args.num_examples, args.max_len, args.split, args.flush_every)


if __name__ == "__main__":
    main()
