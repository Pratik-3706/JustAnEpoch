import os
import sys
from pathlib import Path
import warnings

# Ignore annoying PyTorch warnings
warnings.filterwarnings("ignore")

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.model.transformer import Transformer

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Please pip install tokenizers")
    sys.exit(1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    test_dir = Path(__file__).parent
    pt_files = list(test_dir.glob("*.pt"))
    if not pt_files:
        print("Error: Could not find any .pt checkpoint in the test folder.")
        return
    checkpoint_path = pt_files[0]
    
    tokenizer_path = test_dir / "private_colab_tokenizer.json"
    if not tokenizer_path.exists():
        print("Error: Could not find tokenizer in the test folder.")
        return
        
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    print("Loading model weights...")
    model = Transformer(vocab_size=tokenizer.get_vocab_size(), block_size=256)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    step = checkpoint.get('step', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    print(f"✅ Successfully loaded model from Step {step}")
    
    model.to(device)
    model.eval()
    
    print("\n" + "="*50)
    print("🤖 MODEL READY! Type 'quit' or 'exit' to stop.")
    print("Note: The model is only at 3k steps (Loss ~4), so it will likely speak gibberish!")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            # Format as conversation so the model understands
            prompt = f"<HUMAN> {user_input}\n<GPT> "
            input_ids = tokenizer.encode(prompt).ids
            x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            print("Assistant: ", end="", flush=True)
            
            with torch.no_grad():
                out_ids = model.generate(x, max_new_tokens=50, temperature=0.8)
                
            # Decode just the newly generated tokens
            generated_ids = out_ids[0][len(input_ids):].tolist()
            response = tokenizer.decode(generated_ids)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
