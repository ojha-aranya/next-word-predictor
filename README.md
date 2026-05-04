# Next-Word Predictor

Interactive next-word prediction using deep LSTM, trained on 1.1M sequences with style-conditional generation.

🔗 **[Live Demo](https://next-word-predictor-szftqzphbymsdz3xpuz75k.streamlit.app)**

## Features

- **Two prediction styles**: Normal and Shakespearean
- **Real-time suggestions**: Top-6 next-word predictions with confidence scores
- **Context window**: Maintains last 20 tokens for accurate predictions
- **Interactive UI**: Built with Streamlit

## Model Performance

| Architecture | Test PPL | Test Acc | Best Epoch |
|--------------|----------|----------|------------|
| **LSTM** (winner) | **925.6** | **14.18%** | 19 |
| GRU | 944.2 | 14.12% | 16 |

- Vocab size: 62,549 tokens
- Training data: 1.1M sequences
- Mean sequence length: 5.7 tokens

## Architecture

- 2-layer LSTM with attention pooling
- Embedding dim: 256
- Hidden dim: 512
- Dropout: 0.4
- Gradient clipping: max_norm=1.0
- Label smoothing: 0.1
- Weight decay: 1e-5

## Installation

```bash
git clone https://github.com/ojha-aranya/next-word-predictor
cd next-word-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- PyTorch
- Streamlit
- scikit-learn
- Git LFS (for model weights)

## Training Improvements

Resolved overfitting by implementing:
1. Dropout regularization (0.4)
2. Attention pooling over all LSTM layers
3. Label smoothing (0.1)
4. Weight decay (1e-5)
5. Gradient clipping (max_norm=1.0)
6. LR scheduling with early stopping

Result: Reduced train/test gap from 2.5x to near-zero, stable convergence achieved.