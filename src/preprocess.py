import os
import numpy as np
import argparse
from datasets import load_dataset
from src.tokenizer import Tokenizer


def tokenize_and_save(dataset_name: str,
                      split: str,
                      dataset_type: str,
                      text_fields: dict,
                      tokenizer_path: str,
                      save_dir: str,
                      max_prompt_len: int = 256,
                      max_response_len: int = 256,
                      limit: int = 2_000_000,
                      dtype=np.uint32):
    """
    Tokenize a dataset and save as memmap .dat file.

    For OpenWebText: flattened 1D array [total_n_tokens].
    For instruction/reasoning: 2D array [num_examples, 2, max_len].
    """
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path)

    # Load the dataset
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split=split)

    tokens_path = os.path.join(save_dir, f"{dataset_name.replace('/', '_')}_{split}.dat")

    if dataset_type == "openwebtext":
        # Flattened array
        num_samples = len(dataset)
        mmap = np.memmap(tokens_path, dtype=dtype, mode="w+", shape=(num_samples * 200,))
        cursor = 0

        for example in dataset:
            text = example[text_fields["text"]]
            ids = tokenizer.encode(text)
            n = len(ids)
            if cursor + n > limit:
                break
            mmap[cursor:cursor+n] = ids
            cursor += n

        mmap.flush()
        print(f"Saved {cursor} tokens to {tokens_path}")

    else:
        # Instruction / reasoning: save as 2D array [num_examples, 2, max_len]
        num_examples = len(dataset)
        mmap = np.memmap(tokens_path, dtype=dtype, mode="w+",
                         shape=(num_examples, 2, max(max_prompt_len, max_response_len)))

        for i, example in enumerate(dataset):
            prompt_ids = tokenizer.encode(example[text_fields["prompt"]])
            response_ids = tokenizer.encode(example[text_fields["response"]])

            # pad or truncate
            prompt_ids = (prompt_ids + [tokenizer.pad_id] * max_prompt_len)[:max_prompt_len]
            response_ids = (response_ids + [tokenizer.pad_id] * max_response_len)[:max_response_len]

            mmap[i, 0, :] = prompt_ids
            mmap[i, 1, :] = response_ids

        mmap.flush()
        print(f"Saved {num_examples} examples to {tokens_path}")

    return tokens_path


def main():
    parser = argparse.ArgumentParser(description="Download, tokenize, and save datasets as memmap files.")

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset identifier, e.g., 'openwebtext', 'gsm8k'")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train, validation, etc.")
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["pretraining", "instruction"],
                        help="Type of dataset (affects how data is concatenated, shaped)")
    parser.add_argument("--text_fields", type=str, nargs="+", required=True,
                        help="Text fields to use. For pretraining: text; instruction: prompt response.")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to trained tokenizer model (.model file)")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Directory to save tokenized memmap files")
    parser.add_argument("--max_prompt_len", type=int, required=False, default=256, 
                        help="NumPy dtype for memmap (default: uint32)")
    parser.add_argument("--max_response_len", type=int, required=False, default=256, 
                        help="NumPy dtype for memmap (default: uint32)")
    parser.add_argument("--limit", type=int, required=False, default=2_000_000, 
                        help="NumPy dtype for memmap (default: uint32)")
    parser.add_argument("--dtype", type=str, default="uint32", help="NumPy dtype for memmap (default: uint32)")

    args = parser.parse_args()

    # convert text_fields list into dict
    if args.dataset_type == "pretraining":
        fields = {"text": args.text_fields[0]}
    elif args.dataset_type == "instruction":
        fields = {"prompt": args.text_fields[0], "response": args.text_fields[1]}
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    # run tokenization
    tokenize_and_save(
        dataset_name=args.dataset_name,
        split=args.split,
        dataset_type=args.dataset_type,
        text_fields=fields,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir,
        max_prompt_len=args.max_propt_len,
        max_response_len=args.max_response_len,
        limit=args.limit,
        dtype=getattr(__import__("numpy"), args.dtype)
    )


if __name__ == "__main__":
    main()
