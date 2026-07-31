""" Script to train the BPE tokenizer. """

from pathlib import Path
from src.tokenizer import Tokenizer

def train():
    tokenizer = Tokenizer()
    
    max_bytes_limit = 50 * 1024 * 1024  # 50MB ig lol
    vocab_count = tokenizer.load_dataset_for_bpe(max_bytes=max_bytes_limit)
    
    # Subtract the 3 special tokens so the total vocab is exactly 32000
    for i in range(tokenizer.vocab_size - 256 - len(tokenizer.special_tokens)):
        new_id = 256 + i
        stats = tokenizer.get_stats(vocab_count)
        
        # If there are no more pairs to merge
        if not stats:
            print("No more pairs to merge! Stopping early.")
            break
            
        best_pair = max(stats, key=stats.get)
        tokenizer.merges[best_pair] = new_id
        
        # CRITICAL: You must update the vocab dictionary so it knows how to decode this new ID!
        tokenizer.vocab[new_id] = tokenizer.vocab[best_pair[0]] + tokenizer.vocab[best_pair[1]]
        
        vocab_count = tokenizer.merge_vocab(best_pair, new_id, vocab_count)
        
        # Add a print statement so you aren't staring at a blank screen for an hour
        if i % 100 == 0:
            print(f"Merge {i}: {best_pair} -> {new_id}")
        
    # 1. Define your custom output directory and file path
    output_dir = Path(__file__).resolve().parent.parent / "dataset" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))

if __name__ == "__main__":
    train()
