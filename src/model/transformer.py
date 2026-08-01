"""
Transformer architecture implementation.
"""
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 32003,
        d_model: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size

        # Embeddings 
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        # transformers Encoder layers 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead= n_head,
            dim_feedforward= 4 * d_model,
            dropout= dropout,
            batch_first= True

        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # output 

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embedding.weight = self.head.weight


    # only god knows how its working right now.

    def forward(self, x):
        b, t = x.shape
        assert t <= self.block_size, f"Sequence length {t} exceeds maximum block size {self.block_size}!"
        
        # 1. Compute token and positional embeddings
        tok_emb = self.token_embedding(x)  # Shape: (Batch, Time, d_model)
        pos_emb = self.position_embedding(torch.arange(t, device=x.device))  # Shape: (Time, d_model)
        x_emb = tok_emb + pos_emb

        # 2. Generate causal mask to prevent looking into the future
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
        
        # 3. Pass through transformer layers
        x_out = self.transformer(x_emb, mask=mask, is_causal=True)
        
        # 4. Final normalization and projection to vocabulary logits
        x_out = self.ln_f(x_out)
        logits = self.head(x_out)  # Shape: (Batch, Time, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressively generate new tokens.
        idx is (Batch, Time) array of indices in the current context
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size to prevent crashing
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits = self(idx_cond)
            
            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx