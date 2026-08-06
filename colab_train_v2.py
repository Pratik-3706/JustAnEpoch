"""
Google Colab Training Script V2 — Scaled to ~50M Parameters
This script creates a larger model and transfers learned weights from the
old 22M checkpoint (v2) so training doesn't start from scratch.

Your original colab_train.py and transformer.py are NOT modified.
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
        
        # Check if already mounted to avoid crashing
        if not os.path.exists('/content/drive/MyDrive') and not os.path.exists('/content/drive'):
            print("Attempting to mount Google Drive...")
            try:
                drive.mount('/content/drive')
            except AttributeError:
                print("\n" + "="*60)
                print("⚠️ ERROR: Cannot mount Google Drive from a python script!")
                print("Please create a new Colab cell and run this first:")
                print("from google.colab import drive")
                print("drive.mount('/content/drive')")
                print("="*60 + "\n")
                return False
        else:
            print("Google Drive is already mounted!")
            
        # Add the project root to sys.path so imports work in Colab
        project_root = "/content/JustAnEpoch"
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        return True
    except ImportError:
        print("Not running in Google Colab. Please run src/train.py instead.")
        return False


def transfer_weights_from_old_checkpoint(new_model, old_checkpoint_path, device):
    """
    Transfer learned weights from the old 22M model (d_model=384, n_layer=6)
    into the new 50M model (d_model=512, n_layer=8).
    
    Strategy:
    - For weight matrices where d_model changed (384 → 512):
      Copy old weights into the first 384 dims, initialize the rest small random.
    - For transformer layers: transfer layers 0-5, leave layers 6-7 randomly initialized.
    - For biases and LayerNorms: same padding approach.
    """
    print("=" * 60)
    print("🔄 WEIGHT TRANSFER: Loading old 22M checkpoint...")
    print("=" * 60)
    
    old_checkpoint = torch.load(old_checkpoint_path, map_location=device, weights_only=False)
    old_state = old_checkpoint['model_state_dict']
    new_state = new_model.state_dict()
    
    old_d_model = 384
    new_d_model = 512
    old_n_layer = 6
    
    transferred = 0
    skipped = 0
    
    for name, new_param in new_state.items():
        if name not in old_state:
            # This param doesn't exist in old model (e.g. layers 6-7)
            skipped += 1
            continue
            
        old_param = old_state[name]
        
        if old_param.shape == new_param.shape:
            # Shapes match exactly — direct copy
            new_state[name] = old_param.clone()
            transferred += 1
            
        elif len(old_param.shape) == 2 and len(new_param.shape) == 2:
            # 2D weight matrix — need to pad dimensions
            old_r, old_c = old_param.shape
            new_r, new_c = new_param.shape
            
            # Start with the randomly initialized new weights
            # Copy old weights into the top-left corner
            min_r = min(old_r, new_r)
            min_c = min(old_c, new_c)
            new_state[name][:min_r, :min_c] = old_param[:min_r, :min_c].clone()
            transferred += 1
            
        elif len(old_param.shape) == 1 and len(new_param.shape) == 1:
            # 1D bias or LayerNorm — pad with zeros (bias) or appropriate values
            min_size = min(old_param.shape[0], new_param.shape[0])
            new_state[name][:min_size] = old_param[:min_size].clone()
            transferred += 1
            
        else:
            skipped += 1
    
    new_model.load_state_dict(new_state)
    
    print(f"✅ Transferred {transferred} parameter tensors from old model")
    print(f"⏭️  Skipped {skipped} tensors (new layers or shape mismatch)")
    print(f"Old checkpoint was at epoch {old_checkpoint.get('epoch', '?')}")
    print("=" * 60)
    
    return new_model


def train_colab():
    if not setup_colab():
        return
        
    # We must import these AFTER sys.path is updated in Colab
    from src.dataset.dataset import download_and_save_dataset
    from src.tokenizer import Tokenizer
    from src.tokenizer.train_tokenizer import train as train_tokenizer_script
    from src.dataset.dataloader import TextDataset
    # Import the V2 model instead of the original
    from src.model.transformer_v2 import Transformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # COLAB PATHS (Storing EVERYTHING in Drive for persistence!)
    # -------------------------------------------------------------
    project_root = Path("/content/JustAnEpoch")
    
    # Persistent Drive Directory
    drive_dir = Path("/content/drive/MyDrive/JustAnEpoch_Checkpoints")
    drive_dir.mkdir(parents=True, exist_ok=True)
    
    # V3 checkpoint for the new model — old v2 checkpoint stays untouched
    checkpoint_path = drive_dir / "latest_checkpoint_v3.pt"
    # Old checkpoint for weight transfer
    old_checkpoint_path = drive_dir / "latest_checkpoint_v2.pt"
    
    
    dataset_path = drive_dir / "dataset" / "dataset_train.txt"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Dataset
    if not dataset_path.exists():
        print("Downloading 900MB dataset into Colab...")
        download_and_save_dataset(save_dir=dataset_path.parent)
    
    
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError:
        print("Installing tokenizers...")
        os.system("pip install tokenizers")
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    hf_tokenizer_path = drive_dir / "private_colab_tokenizer.json"
    
    if not hf_tokenizer_path.exists():
        print("Training a HuggingFace Tokenizer on full 1GB dataset...")
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
    
    # 4. DataLoader — batch_size=32 for comfortable VRAM on T4
    print("Encoding Dataset...")
    block_size = 256
    dataset = TextDataset(filepath=str(dataset_path), tokenizer=tokenizer, max_length=block_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 5. Model — V2 with ~50M parameters (d_model=512, n_head=8, n_layer=8)
    model = Transformer(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model.to(device)
    
    # Count and display parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model V2 initialized with {total_params:,} parameters (~{total_params/1e6:.1f}M)")
    
    # 6. Optimizer & AMP & Scheduler
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 10
    total_steps = epochs * len(dataloader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.05)
    
    # 7. Checkpoint Loading
    start_epoch = 0
    
    if checkpoint_path.exists():
        # Resume from V3 checkpoint (this model's own checkpoint)
        print(f"Found V3 checkpoint in Google Drive! Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch']
        
        # If the epoch was fully completed, start the next one
        if checkpoint.get('epoch_completed', False):
            start_epoch += 1
            
        print(f"Resuming from epoch {start_epoch}")
        
    elif old_checkpoint_path.exists():
        # First run of V2 — transfer weights from old 22M model
        print("No V3 checkpoint found. Transferring weights from old 22M model...")
        model = transfer_weights_from_old_checkpoint(model, old_checkpoint_path, device)
        print("Weight transfer complete! Starting training from epoch 0.")
    else:
        print("No checkpoints found. Starting from scratch with random initialization.")
        
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
            
            # Forward pass with Mixed Precision
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))
            
            # Backward pass with AMP and Gradient Clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f" [Mid-Epoch Checkpoint saved to Drive at step {step+1}]")
            
        # End of Epoch Save
        torch.save({
            'epoch': epoch,
            'epoch_completed': True,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': total_loss / len(dataloader),
        }, checkpoint_path)
        print(f"Epoch {epoch} complete! Checkpoint saved to Drive.")

if __name__ == "__main__":
    train_colab()
