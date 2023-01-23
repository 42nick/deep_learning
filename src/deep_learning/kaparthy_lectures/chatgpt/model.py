import torch
import torch.nn as nn
from torch.nn import functional as F


class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits = self.token_embedding(idx)  # (B, T, C) batches, timestamps (length of the input sequence), vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # logits have to be flattened to (B*T, C)
            logits = logits.view(-1, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return loss, logits

    def predict(self, idx: torch.Tensor, additional_predicted_chars: int) -> torch.Tensor:
        # idx is of shape (B, T)
        for _ in range(additional_predicted_chars):
            # get the predictions
            _, logits = self.forward(idx, None)
            # focus on the last time step only
            logits = logits[:, -1, :]
            # apply softmax to get the probabilities
            props = F.softmax(logits, dim=-1)  # (B, C)
            # getting one token from the distribution
            idx_next = torch.multinomial(props, num_samples=1)  # (B, 1)
            # append the new token to the input
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
