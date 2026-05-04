import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model Architecture (must match training) ----

class DeepLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_indx, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_indx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.4)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        hidden_t = hidden.permute(1, 0, 2)
        weights = torch.softmax(self.attn_pool(hidden_t), dim=1)
        context = (weights * hidden_t).sum(dim=1)
        context = self.layer_norm(context)
        context = self.dropout(context)
        return self.linear(context)


# ---- Load tokenizer and model ----

def load_tokenizer(path="tokenizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model(path="lstm_model_best.pth", vocab_size=62550,
               emb_dim=256, hidden_dim=512, num_layers=2):
    model = DeepLSTMModel(vocab_size, emb_dim, hidden_dim, 0, num_layers)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model


# ---- Core inference function ----

def predict_next_words(
    token_sequence,     # list of int token indices already built up
    model,
    tokenizer,
    top_k=4,
    max_len=20,         # max context window
    style_token=2,      # 1=shakespeare, 2=normal
):
    # Enforce max context: keep style token + last (max_len - 1) words
    if len(token_sequence) > max_len:
        token_sequence = [style_token] + token_sequence[-(max_len - 1):]

    src = torch.tensor([token_sequence]).to(device)
    lengths = torch.tensor([len(token_sequence)])

    with torch.no_grad():
        logits = model(src, lengths)
        probs = F.softmax(logits[0], dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for idx, prob in zip(top_indices, top_probs):
        word = tokenizer.index_word.get(idx.item(), "<UNK>")
        if word not in ("<shakespeare>", "<normal>", "<UNK>"):
            results.append((word, round(prob.item() * 100, 2)))

    return results


def build_token_sequence(style_token, words, tokenizer):
    """Convert list of word strings into token index list with style token prepended."""
    indices = []
    for w in words:
        idx = tokenizer.word_index.get(w.lower())
        if idx is not None:
            indices.append(idx)
        # Unknown words are silently skipped — model only knows its vocab
    return [style_token] + indices