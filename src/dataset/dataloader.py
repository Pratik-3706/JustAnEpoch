"""
PyTorch Dataset and DataLoader for the text data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import array

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
            print("Tokenizing dataset (this is CPU intensive)...")
            tokens = array.array('i')
            with open(filepath, "r", encoding="utf-8") as f:
                chunk = []
                lines_processed = 0
                for line in f:
                    chunk.append(line)
                    lines_processed += 1
                    
                    if len(chunk) >= 5000:
                        text = "".join(chunk)
                        tokens.extend(self.tokenizer.encode(text))
                        chunk = []
                        if lines_processed % 100000 == 0:
                            print(f"Processed {lines_processed} lines...")
                            
                # Process remaining lines
                if chunk:
                    tokens.extend(self.tokenizer.encode("".join(chunk)))
            
            print("Tokenization complete! Converting to PyTorch Tensor...")
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            print(f"Saving 1.4GB tensor to Google Drive at {cache_path}...")
            torch.save(tokens, cache_path)
            print("Save complete!")

        print("Building sequence examples...")
        self.examples = []
        for i in range (0, len(tokens)- max_length, max_length):
            chunk = tokens[i : i + max_length +1 ]
            if len(chunk) == max_length + 1:
                self.examples.append(chunk)
        print("Dataset loaded successfully!")

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        chunk = self.examples[idx]

        x = torch.as_tensor(chunk[:-1], dtype=torch.long)
        y = torch.as_tensor(chunk[1:], dtype=torch.long)
        return x, y