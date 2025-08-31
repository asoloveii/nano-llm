import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import torch 
from torch.nn.utils.rnn import pad_sequence
import regex as re
import numpy as np
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
    def train(cls, corpus_path: str, output_dir: str, vocab_size: int = 32768,
              model_type: str = "bpe", character_coverage: float = 1.0):
        """
        Train a SentencePiece tokenizer and return a loaded instance.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_prefix = Path(output_dir) / "bpe_tokenizer"

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )

        # load and return the tokenizer instance
        return cls(str(model_prefix) + ".model")


class BPETokenizer:

    def __init__(self, 
                 corpus: os.PathLike | str, 
                 vocab_size: int, 
                 special_tokens: List[str] = None):
        '''
        Byte-Pair Encoder(BPE) tokenizer. 
        This is a custom, but inefficient implementation.
        '''
        # pre-tokenizing regex string
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # initialize variables
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = defaultdict(int)
        self.vocab_size: int = vocab_size
        self.special_tokens: Dict[str, int] = {} 
        self.pad_token_idx: int = 0

        # train bpe on corpus
        self._create_vocab(corpus, special_tokens)

    def _create_vocab(self, corpus: os.PathLike | str, special_tokens):
        '''Trains a vocabulary for a given corpus'''
        # initial vocabulary of 256 UTF-8 tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # add special tokens to the vocabulary
        next_idx = len(self.vocab)
        for token in special_tokens:
            if token == '<PAD>':
                self.pad_token_idx = next_idx
            self.vocab[next_idx] = token.encode("utf-8")
            self.special_tokens[token] = next_idx
            next_idx += 1

        # number of merges needed to be done
        n_merges = self.vocab_size - len(self.vocab)

        # save the text from corpus into a string
        if isinstance(corpus, os.PathLike):
            with open(corpus, "r") as file:
                text = file.read()
        else:
            text = corpus

        # pretokenize text and convert each pretoken into array of baic UTF-8 tokens
        pretokenized = re.findall(self.PAT, text)
        ids = [np.frombuffer(token.encode("utf-8"), dtype=np.uint8) 
                for token in pretokenized]

        next_idx = len(self.vocab)
        while n_merges != 0:
            # collect stats of pairs of tokens
            stats = defaultdict(int)
            for chunk in ids:
                chunk_stats = self.get_stats(chunk)
                for pair, count in chunk_stats.items():
                    stats[pair] += count 

            if not stats:
                break
            
            # get the most frequent pair
            pair = max(stats, key=stats.get)
            # merge the most frequent pair
            ids = [self.merge(chunk, pair, next_idx) for chunk in ids]

            # save new merge
            self.vocab[next_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = next_idx
            # update loop variants
            next_idx += 1
            n_merges -= 1

    def get_stats(self, chunk: np.ndarray):
        '''
        Given an array chunk of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        '''
        counts = {} 
        for pair in zip(chunk, chunk[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, chunk: np.ndarray, pair: Tuple[int, int], idx: int):
        '''
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        '''
        newids = []
        i = 0
        while i < len(chunk):
            # if not at the very last position AND the pair matches, replace it
            if chunk[i] == pair[0] and i < len(chunk) - 1 and chunk[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(chunk[i])
                i += 1
        return np.array(newids)

    def encode(self, text: str) -> List[int]:
        '''Encode an input text into a sequence of token IDs.'''
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        return ids 

    def decode(self, ids: List[int]) -> str:
        '''Decode a sequence of token IDs into text.'''
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text  
    
    def pad_sequences(self, sequences: List[List[int]], max_seq_len: int) -> List[int]:
        # convert to tensors first
        seq_tensors = [torch.tensor(seq[:max_seq_len], dtype=torch.long) for seq in sequences]

        # pads to longest seq by default
        padded = pad_sequence(seq_tensors, batch_first=True, padding_value=self.pad_token_idx)

        # if some sequences shorter than max_seq_len, pad_sequence won’t extend further
        if padded.size(1) < max_seq_len:
            extra_pad = torch.full(
                (padded.size(0), max_seq_len - padded.size(1)), 
                self.pad_token_idx, 
                dtype=torch.long
            )
            padded = torch.cat([padded, extra_pad], dim=1)

        # truncate sequences longes than max_seq_len
        if padded.size(1) > max_seq_len:
            padded = padded[:, :max_seq_len]

        return padded

    @classmethod
    def load_files(cls, 
                   vocab_path: os.PathLike, 
                   merges_path: os.PathLike, 
                   special_tokens: List[str]):
        '''Load pretrained vocabulary and merges.'''
         # load vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # convert keys back to int and values back to bytes
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        # load merges
        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges = {tuple(map(int, k.split(","))): v for k, v in raw_merges.items()}

        # determine vocab size
        vocab_size = len(vocab)

        # construct tokenizer
        tokenizer = cls.__new__(cls)  # create instance without calling __init__
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.vocab_size = vocab_size
        tokenizer.special_tokens = {}
        tokenizer.pad_token_idx = 0
        tokenizer.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # set special tokens if provided
        if special_tokens:
            for token in special_tokens:
                for idx, val in vocab.items():
                    if val == token.encode("utf-8"):
                        tokenizer.special_tokens[token] = idx
                        if token == "<PAD>":
                            tokenizer.pad_token_idx = idx

        return tokenizer  

    def save(self, vocab_path: os.PathLike, merges_path: os.PathLike ):
        '''Save vocab and merges to JSON files.'''
        # convert vocab (bytes -> hex string) for safe JSON serialization
        vocab_serializable = {str(k): v.hex() for k, v in self.vocab.items()}

        # convert merges (tuple -> "a,b" string)
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
