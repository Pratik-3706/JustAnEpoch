# StupidLM

A simple, from-scratch implementation of a GPT-style Transformer language model and Byte-Pair Encoding (BPE) Tokenizer in pure Python and PyTorch.

> **Note:** This repository was built from the ground up from first principles. It's just for my understanding only!

## Project Structure
- `src/model/transformer.py`: The core Transformer architecture using causal masking and weight-tied embeddings.
- `src/tokenizer/tokenizer.py`: A from-scratch, pure Python implementation of a Byte-Pair Encoding (BPE) tokenizer.
- `src/dataset/dataloader.py`: A PyTorch `Dataset` that streams text, encodes it, and yields sequence batches.
- `src/train.py`: The main PyTorch training loop for local execution.
- `colab_train.py`: A training script for Google Colab (uses HuggingFace for faster tokenization and saves checkpoints to Google Drive).

## Credits
Built as a personal learning project to deeply understand the mechanics of Large Language Models (LLMs), tokenization algorithms, and autoregressive training loops.
