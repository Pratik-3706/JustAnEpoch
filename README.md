# StupidLM

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-StupidLm-yellow)](https://huggingface.co/Pratik-3706/StupidLm)

A simple, from-scratch implementation of a GPT-style Transformer language model and Byte-Pair Encoding (BPE) Tokenizer in pure Python and PyTorch.

> **Note:** This repository was built from the ground up from first principles. It's just for my understanding only!

### Pretrained Weights
If for some reason you want to download a 48M parameter model that generates confident gibberish, the weights are available on Hugging Face: **[Pratik-3706/StupidLm](https://huggingface.co/Pratik-3706/StupidLm)**. Don't say I didn't warn you.

## Project Structure
- `src/model/transformer.py`: The core Transformer architecture using causal masking and weight-tied embeddings.
- `src/tokenizer/tokenizer.py`: A from-scratch, pure Python implementation of a Byte-Pair Encoding (BPE) tokenizer.
- `src/dataset/dataloader.py`: A PyTorch `Dataset` that streams text, encodes it, and yields sequence batches.
- `src/train.py`: The main PyTorch training loop for local execution.
- `colab_train.py`: A training script for Google Colab (uses HuggingFace for faster tokenization and saves checkpoints to Google Drive).

## Objective
Built as a personal learning project to deeply understand the mechanics of Large Language Models (LLMs), tokenization algorithms, and autoregressive training loops.

## Credits & Acknowledgements
- **OpenAI:** For the original GPT-2 architecture and Byte-Pair Encoding algorithm that this project implements from scratch.
- **Teknium:** For the excellent `OpenHermes-2.5` dataset used to train the reasoning capabilities of this model.
- **PyTorch & HuggingFace:** For the backend hardware acceleration (`nn.TransformerEncoder` SDPA) and the blazing fast Rust tokenization engine used in Colab.
