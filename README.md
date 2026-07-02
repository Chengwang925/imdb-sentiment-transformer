# SentimentScope: Transformer-Based Sentiment Analysis from Scratch

A transformer model built **from scratch in PyTorch** (no pre-trained weights) that classifies IMDB movie reviews as positive or negative, achieving **78% test accuracy** and surpassing the 75% project target.

## Project Overview

SentimentScope simulates the work of a Machine Learning Engineer at CineScope, an entertainment company improving its recommendation system by understanding user sentiment. The goal: build and train a transformer-based classifier on 50,000 IMDB movie reviews.

Unlike typical fine-tuning projects, this model's architecture (attention heads, transformer blocks, classification head) is implemented **entirely from scratch**. Only the `bert-base-uncased` tokenizer is borrowed from Hugging Face for subword tokenization.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **78.03%** |
| Validation accuracy (final) | ~79% |
| Project target | 75% |

Validation accuracy per epoch: 73.6% → 77.8% → ~79%, showing steady convergence over 5 epochs.

## Model Architecture

A GPT-style transformer adapted for binary classification:

- **Token + positional embeddings** (learned)
- **8 stacked transformer blocks**, each with:
  - Multi-head self-attention (8 heads × head size 32 = 256 dims)
  - Feed-forward network (4× expansion, GELU activation)
  - Pre-LayerNorm residual connections
- **Mean pooling** across the sequence dimension
- **Linear classification head** (2 classes)

Key hyperparameters:

```python
config = {
    "d_embed": 256,
    "context_size": 128,
    "layers_num": 8,
    "heads_num": 8,
    "head_size": 32,
    "dropout_rate": 0.15,
}
```

Training setup: AdamW optimizer (lr = 2e-4, weight decay = 0.01), cross-entropy loss, batch size 32, 5 epochs on GPU.

## Pipeline

1. **Data loading**: Read 50,000 raw `.txt` reviews from the aclImdb directory structure into Pandas DataFrames
2. **Exploration**: Label distribution, review length analysis (characters and words), sample inspection
3. **Preparation**: Shuffle and split into train (22,500) / validation (2,500) / test (25,000)
4. **Tokenization**: `bert-base-uncased` subword tokenizer, truncation/padding to 128 tokens
5. **Data loading**: Custom `IMDBDataset` class + PyTorch `DataLoader` for batching
6. **Model**: Custom `DemoGPT` transformer with classification head
7. **Training**: Epoch-based loop with per-epoch validation accuracy tracking
8. **Evaluation**: Accuracy on the held-out 25,000-review test set

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Dataset

Download the [Large Movie Review Dataset (aclImdb v1)](https://ai.stanford.edu/~amaas/data/sentiment/) and place `aclImdb_v1.tar.gz` in the project root. The notebook extracts it automatically.

### Run

Open `SentimentScope.ipynb` and run the cells in order. A GPU is strongly recommended for training (about 5 epochs × 704 steps).

## Experiments

Three configurations were compared:

| Config | d_embed | Layers | Heads | Dropout | Result |
|---|---|---|---|---|---|
| Baseline | 128 | 4 | 4 | 0.10 | Lower accuracy |
| Medium | 256 | 6 | 8 | 0.10 | ~79% validation |
| **Final** | **256** | **8** | **8** | **0.15** | **78% test** |

Increasing embedding size and depth, combined with slightly stronger dropout and weight decay, gave the best generalization.

## Key Takeaways

- A custom transformer, even without pre-trained weights, can exceed 75% accuracy on binary sentiment classification
- Subword tokenization with consistent padding/truncation is essential for clean transformer inputs
- Mean pooling + a linear head is a simple, effective way to adapt a sequence model for classification
- Per-epoch validation tracking guided hyperparameter choices (epochs, learning rate, capacity)

## Possible Extensions

- Use attention masks to exclude padding tokens from attention
- Try CLS-token pooling instead of mean pooling
- Fine-tune a fully pre-trained model (BERT/RoBERTa) for comparison
- Hyperparameter search over learning rate, warmup, and longer context

## Tech Stack

Python · PyTorch · Hugging Face Transformers · Pandas · Matplotlib
