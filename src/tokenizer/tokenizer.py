import collections
from pathlib import Path
import re
import json

class Tokenizer:
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_DATASET_PATH = (
        SCRIPT_DIR.parent
        / "dataset"
        / "reasoning_dataset"
        / "dataset_train.txt"
    )

    def __init__(self, vocab_size: int = 32000, dataset_path: Path = None):
        self.vocab_size = vocab_size
        self.dataset_path = dataset_path or self.DEFAULT_DATASET_PATH
        
        # 1. Define your special tags
        self.special_tokens = {
            "<HUMAN>": 32000,
            "<GPT>": 32001,
            "<|endoftext|>": 32002
        }
        
        # 2. Update regex to look for special tags FIRST
        special_pattern = "|".join(re.escape(k) for k in self.special_tokens.keys())
        self.gpt_pattern = re.compile(f"(?:{special_pattern})|\\w+|\\s+|[^\\w\\s]")
        
        self.merges = {}  
        
        # 3. Base 256 bytes + add special tokens directly to vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str.encode("utf-8")

    def load_dataset_for_bpe(self, max_bytes: int = None):
        """Streams a large file and builds a frequency table of raw UTF-8 bytes."""
        vocab_counts = collections.defaultdict(int)
        bytes_read = 0

        print(f"Loading {self.dataset_path} line-by-line...")

        with open(
            self.dataset_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            for line in f:
                if max_bytes is not None and bytes_read > max_bytes:
                    break
                bytes_read += len(line.encode("utf-8"))
                
                # Split line into words/punctuation
                words = self.gpt_pattern.findall(line)
                for word in words:
                    # Convert string word to a tuple of raw UTF-8 byte integers
                    byte_tuple = tuple(word.encode("utf-8"))
                    vocab_counts[byte_tuple] += 1

        print(
            f"Done! Processed file into {len(vocab_counts):,} unique word chunks."
        )
        return vocab_counts

    #----- 3 BEP Traning Helpers --------------------
    def get_stats(self, vocab_count: dict) -> dict:
        counts = {}
        #loops through every unique word and its freq 
        for word, freq in vocab_count.items():
            #zip (words, word [1:]) creates adjcent pairs ex: word = (104, 101, 108) ==> pairs(104, 101) and (101, 108)
            for pair in zip(word, word[1:]):
                counts[pair]=counts.get(pair, 0) + freq  # 3. Add the word's frequency to this pair's total count
        return counts 

    def merge_vocab(self, pair: tuple, new_id: int, vocab_counts: dict) -> dict:
        #replace all consecutive occurence of pair with the new token idx
        new_vocab = {}
        
        for word , count in vocab_counts.items():
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(new_id)
                    i += 2
                else: 
                    new_word.append(word[i])
                    i += 1

            new_vocab[tuple(new_word)] = count

        return new_vocab

    # --- 4. ENCODING & DECODING ---
    def encode(self, text: str) -> list[int]:
        words = self.gpt_pattern.findall(text)
        ids = []
        
        for word in words:
            # CHECK FOR SPECIAL TOKEN FIRST
            if word in self.special_tokens:
                ids.append(self.special_tokens[word])
                continue # Skip the BPE merging for this word!
                
            word_ids = list(word.encode("utf-8"))

            while len(word_ids) >= 2:
                pairs = [(word_ids[i], word_ids[i+1]) for i in range(len(word_ids) - 1)]
                best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if best_pair not in self.merges:
                    break

                new_id = self.merges[best_pair]
                new_word_ids = []
                i = 0
                while i < len(word_ids):
                    if i < len(word_ids) - 1 and word_ids[i] == best_pair[0] and word_ids[i+1] == best_pair[1]:
                        new_word_ids.append(new_id)
                        i += 2
                    else:
                        new_word_ids.append(word_ids[i])
                        i += 1
                
                word_ids = new_word_ids
            ids.extend(word_ids)

        return ids
                    
    def decode(self, ids: list[int]) -> str:
        # 1. Look up the bytes for every ID
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)

        # 2. Decode bytes back to a UTF-8 string.
        # errors="replace" prevents crashes if an LLM generates a broken partial byte-token!
        return raw_bytes.decode("utf-8", errors="replace")

    # --- 5. PERSISTENCE ---

    def save(self, filepath: str = "tokenizer.json"):
        # Convert dictionary to a JSON-friendly list of lists
        merges_list = [[p0, p1, new_id] for (p0, p1), new_id in self.merges.items()]
        
        state = {
            "vocab_size": self.vocab_size,
            "merges": merges_list
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"tokenizer saved to {filepath}")


    def load(self, filepath: str = "tokenizer.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.vocab_size = state["vocab_size"]
        self.merges = {(p0, p1): new_id for p0, p1, new_id in state["merges"]}
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
        # Re-inject special tokens after loading merges
        self.special_tokens = {
            "<HUMAN>": 32000,
            "<GPT>": 32001,
            "<|endoftext|>": 32002
        }
        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str.encode("utf-8")
            
        print(f"Tokenizer loaded successfully from {filepath}")