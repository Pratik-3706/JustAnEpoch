"""
Google Colab Training Script for JustAnEpoch
This script is designed to run in Google Colab. It mounts Google Drive,
loads the dataset, and saves checkpoints directly to your Drive so you don't 
lose progress if Colab disconnects or times out.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

def setup_colab():
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Add the project root to sys.path so imports work in Colab
        project_root = "/content/JustAnEpoch"
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        return True
    except ImportError:
        print("Not running in Google Colab. Please run src/train.py instead.")
        return False

def train_colab():
    if not setup_colab():
        return
        
    # We must import these AFTER sys.path is updated in Colab
    from src.dataset.dataset import download_and_save_dataset
    from src.tokenizer import Tokenizer
    from src.tokenizer.train_tokenizer import train as train_tokenizer_script
    from src.dataset.dataloader import TextDataset
    from src.model.transformer import Transformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # COLAB PATHS (Storing EVERYTHING in Drive for persistence!)
    # -------------------------------------------------------------
    project_root = Path("/content/JustAnEpoch")
    
    # Persistent Drive Directory
    drive_dir = Path("/content/drive/MyDrive/JustAnEpoch_Checkpoints")
    drive_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = drive_dir / "latest_checkpoint.pt"
    
    # We will also store the dataset in Drive so you don't have to redownload 1GB every time!
    dataset_path = drive_dir / "dataset" / "dataset_train.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Dataset
    if not dataset_path.exists():
        print("Downloading 900MB dataset into Colab...")
        download_and_save_dataset(save_dir=dataset_path.parent)
    
    # 2 & 3. Train a PRIVATE HuggingFace Tokenizer (Super fast, 100% Offline)
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError:
        print("Installing tokenizers...")
        os.system("pip install tokenizers")
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    hf_tokenizer_path = drive_dir / "private_colab_tokenizer.json"
    
    if not hf_tokenizer_path.exists():
        print("Training a 100% PRIVATE HuggingFace Tokenizer on your full 1GB dataset...")
        print("This runs locally on the Colab machine. NO DATA is uploaded to HuggingFace!")
        
        raw_tokenizer = Tokenizer(models.BPE())
        raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Train to exactly 32000 tokens just like our custom python one
        trainer = trainers.BpeTrainer(
            vocab_size=32000, 
            special_tokens=["<HUMAN>", "<GPT>", "<|endoftext|>"]
        )
        
        # Train directly on the 1GB text file (Takes ~2 minutes in Rust)
        raw_tokenizer.train([str(dataset_path)], trainer=trainer)
        raw_tokenizer.decoder = decoders.ByteLevel()
        raw_tokenizer.save(str(hf_tokenizer_path))
        print(f"Tokenizer securely saved to {hf_tokenizer_path}")
        
    # Create a quick wrapper so your TextDataset thinks it's your custom python tokenizer!
    class PrivateColabTokenizer:
        def __init__(self):
            self.tokenizer = Tokenizer.from_file(str(hf_tokenizer_path))
            self.vocab_size = self.tokenizer.get_vocab_size()
            
        def encode(self, text):
            # .ids returns the raw list of integers, exactly what your dataloader expects!
            return self.tokenizer.encode(text).ids
            
    tokenizer = PrivateColabTokenizer()
    
    # 4. DataLoader
    print("Encoding Dataset...")
    block_size = 256
    dataset = TextDataset(filepath=str(dataset_path), tokenizer=tokenizer, max_length=block_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True) # Larger batch size for Colab GPUs
    
    # 5. Model
    model = Transformer(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model.to(device)
    
    # 6. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 7. Checkpoint Loading from Google Drive
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Found persistent checkpoint in Google Drive! Loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # If the epoch was fully completed, start the next one
        if checkpoint.get('epoch_completed', False):
            start_epoch += 1
            
        print(f"Resuming from epoch {start_epoch}")
        
    # 8. Training Loop
    epochs = 10
    model.train()
    
    # Colab timeouts happen, so we save VERY frequently (e.g. every 1000 steps)
    SAVE_EVERY_N_STEPS = 1000 
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs-1}")
        
        for step, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # COLAB TIMEOUT PROTECTION: Save mid-epoch to Google Drive
            if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'epoch_completed': False,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f" [Mid-Epoch Checkpoint saved to Drive at step {step+1}]")
            
        # End of Epoch Save
        torch.save({
            'epoch': epoch,
            'epoch_completed': True,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
        }, checkpoint_path)
        print(f"Epoch {epoch} complete! Checkpoint saved to Drive.")

if __name__ == "__main__":
    train_colab()
