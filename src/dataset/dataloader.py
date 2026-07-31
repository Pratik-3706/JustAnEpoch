"""
PyTorch Dataset and DataLoader for the text data.
"""
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        from pathlib import Path
        cache_path = Path(filepath).with_suffix('.pt')
        
        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path}...")
            tokens = torch.load(cache_path)
        else:
            print("Tokenizing dataset line-by-line to save RAM (this will take a few minutes)...")
            tokens = []
            with open(filepath, "r", encoding="utf-8") as f:
                chunk = []
                for line in f:
                    chunk.append(line)
                    # Process in chunks of 10,000 lines to maximize Rust tokenization speed
                    # while keeping RAM usage very low.
                    if len(chunk) >= 10000:
                        text = "".join(chunk)
                        tokens.extend(self.tokenizer.encode(text))
                        chunk = []
                # Process remaining lines
                if chunk:
                    tokens.extend(self.tokenizer.encode("".join(chunk)))
            
            print(f"Saving tokenized cache to {cache_path} for instant loading next time!")
            torch.save(tokens, cache_path)

        self.examples = []
        for i in range (0, len(tokens)- max_length, max_length):
            chunk = tokens[i : i + max_length +1 ]
            if len(chunk) == max_length + 1:
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        chunk = self.examples[idx]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y