import argparse
import tempfile
from pathlib import Path
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import sentencepiece as spm


class Tokenizer:

    def __init__(self, model_path: str):
        '''SentencePiece tokenizer.'''
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        # special tokens
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.sep_id = self.sp.piece_to_id("<sep>")

    def encode(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        # add tokens BOS and EOS if needed
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]):
        return self.sp.decode(ids)

    def pad_sequences(self, 
                      sequences: List[List[int]], 
                      max_seq_len: int) -> torch.Tensor:
        '''
        Pads a list of sequences into a tensor, 
        with each sequence's length equal to `max_seq_len`
        '''
        seq_tensors = [torch.tensor(seq[:max_seq_len], dtype=torch.long) 
                       for seq in sequences]
        padded = pad_sequence(seq_tensors, batch_first=True, padding_value=self.pad_id)
        if padded.size(1) < max_seq_len:
            extra_pad = torch.full(
                (padded.size(0), max_seq_len - padded.size(1)),
                self.pad_id, dtype=torch.long
            )
            padded = torch.cat([padded, extra_pad], dim=1)
        return padded[:, :max_seq_len]

    @classmethod
    def train(cls, corpus_path: str, output_dir: str, 
              vocab_size: int, model_type: str, 
              model_name: str, dataset_name: str, split: str,
              character_coverage: float = 1.0):
        '''
        Train a SentencePiece tokenizer and return a loaded instance.
        '''
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_prefix = Path(output_dir) / model_name

        if dataset_name is not None:
            print("Loading dataset...")
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            # write to temporary text file for SentencePiece
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as tmpfile:
                for example in dataset:
                    # assuming structure just like in OWT
                    text = example.get("text") or example.get("plain_text") or ""
                    tmpfile.write(text.replace("\n", " ") + "\n")
                corpus_path = tmpfile.name
            print(f"Dataset written to temporary file {corpus_path}")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            user_defined_symbols=["<sep>", "0","1","2","3","4","5","6","7","8","9", 
                                  "+","-","=","*","/","(",")"]
        )

        # load and return the tokenizer instance
        return cls(str(model_prefix) + ".model")


def main():
    parser = argparse.ArgumentParser(description="Train or use a SentencePiece tokenizer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new tokenizer")
    train_parser.add_argument("--corpus", type=str, default=None,
                              help="Path to corpus .txt file for training")
    train_parser.add_argument("--output_dir", type=str, required=True,
                              help="Directory to save the tokenizer model and vocab")
    train_parser.add_argument("--vocab_size", type=int, default=50304,
                              help="Vocabulary size (default: 50304)")
    train_parser.add_argument("--model_type", type=str, default="bpe",
                              choices=["bpe", "unigram", "char", "word"],
                              help="SentencePiece model type")
    train_parser.add_argument("--model_name", type=str, default="tokenizer",
                              help="Base name for model files")
    train_parser.add_argument("--dataset_name", type=str, required=True,
                              help="HuggingFace dataset identifier")
    train_parser.add_argument("--split", type=str, default="train",
                              help="Dataset split to be used")
    train_parser.add_argument("--character_coverage", type=float, default=1.0,
                              help="Amount of characters covered (1.0 for English, 0.9995 for multilingual)")

    args = parser.parse_args()

    if args.command == "train":
        Tokenizer.train(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            character_coverage=args.character_coverage,
        )
        print(f"Tokenizer trained and saved to {args.output_dir}")


if __name__ == "__main__":
    '''
    python src/tokenizer.py train \
    --output_dir checkpoints/tokenizer \
    --vocab_size 50304 \
    --model_type bpe \
    --dataset_name Elriggs/openwebtext-100k \
    --split train

    '''
    main()
