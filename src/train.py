"""
Main training script for JustAnEpoch model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.dataset.dataset import download_and_save_dataset
from src.tokenizer import Tokenizer
from src.tokenizer.train_tokenizer import train as train_tokenizer_script
from src.dataset.dataloader import TextDataset
from src.model.transformer import Transformer

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths setup
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "src" / "dataset" / "reasoning_dataset" / "dataset_train.txt"
    tokenizer_path = project_root / "src" / "dataset" / "processed" / "tokenizer.json"
    
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
    
    # 1. Download Dataset if it doesn't exist
    if not dataset_path.exists():
        print("Dataset not found. Downloading 900MB dataset...")
        download_and_save_dataset(save_dir=dataset_path.parent)
    
    # 2. Train Tokenizer on 50MB if it doesn't exist
    if not tokenizer_path.exists():
        print("Tokenizer not trained. Training on 50MB chunk...")
        train_tokenizer_script()
        
    # 3. Initialize and Load Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(str(tokenizer_path))
    
    # 4. Load Dataset and DataLoader
    print("Loading and encoding full dataset (this takes some RAM and time!)...")
    block_size = 256
    dataset = TextDataset(filepath=str(dataset_path), tokenizer=tokenizer, max_length=block_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 5. Initialize Transformer Model
    model = Transformer(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model.to(device)
    
    # 6. Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 7. Resume from Checkpoint if exists
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"Found checkpoint! Loading from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        
    # 8. Main Training Loop
    epochs = 10
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs-1}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropyLoss: (Batch * Time, Vocab_Size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)
            
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # (Optional) Frequent Checkpointing inside the epoch every 5000 steps
            if (batch_idx + 1) % 5000 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{batch_idx}.pt")
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} finished with average loss: {avg_loss:.4f}")
        
        # Save End-of-Epoch Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Epoch {epoch} checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
