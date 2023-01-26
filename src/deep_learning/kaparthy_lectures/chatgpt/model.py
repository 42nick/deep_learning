import torch
import torch.nn as nn
from torch.nn import functional as F

DROPOUT_RATIO = 0.2


class SelfAttentionHead(nn.Module):
    """Single head of the self-attention layer."""

    def __init__(self, head_size: int, n_embeddings: int, block_size: int) -> None:
        super().__init__()

        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)

        self.dropout = nn.Dropout(DROPOUT_RATIO)

        # mask is a buffer, not a parameter so it is not updated during training
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        weight = q @ k.transpose(-2, -1) / (C**0.5)  # (B, T, T)
        weight = weight.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)  # (B, T, T)

        weight = self.dropout(weight)

        output = weight @ v  # (B, T, head_size)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embeddings: int, block_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    n_embeddings=n_embeddings,
                    head_size=head_size,
                    block_size=block_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.project = nn.Linear(num_heads * head_size, n_embeddings)
        self.dropout = nn.Dropout(DROPOUT_RATIO)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embeddings: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # factor 4 is used to increase
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(DROPOUT_RATIO),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embeddings: int, block_size: int, num_heads: int) -> None:
        super().__init__()

        head_size = n_embeddings // num_heads

        self.self_attention_heads = MultiHeadAttention(
            num_heads=num_heads, head_size=head_size, n_embeddings=n_embeddings, block_size=block_size
        )
        self.feed_forward = FeedForward(n_embeddings)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class BiagramLanguageModel(nn.Module):
    def __init__(
        self, vocab_size: int, n_embeddings: int, block_size: int, head_size: int, num_heads: int, n_layer: int
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embeddings)
        self.positional_embedding = nn.Embedding(block_size, n_embeddings)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(n_embeddings=n_embeddings, block_size=block_size, num_heads=num_heads)
                for _ in range(n_layer)
            ]
        )

        self.ln_final = nn.LayerNorm(n_embeddings)

        # used to map the embeddings to the vocab_size
        self.output_head = nn.Linear(n_embeddings, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # (B, T, C) batches, timestamps (length of the input sequence), vocab_size
        token_embedding = self.token_embedding(idx)
        positional_embedding = self.positional_embedding(torch.arange(idx.shape[1], device=idx.device))  # (T, C)
        x = token_embedding + positional_embedding  # (B, T, C)
        x = self.transformer_blocks(x)
        x = self.ln_final(x)
        logits = self.output_head(x)  # (B, T, vocab_size)

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
            # only a fixed, limited lenght of indices can be processed at once
            indices = idx[:, -self.block_size :]
            # get the predictions
            _, logits = self.forward(indices, None)
            # focus on the last time step only
            logits = logits[:, -1, :]
            # apply softmax to get the probabilities
            props = F.softmax(logits, dim=-1)  # (B, C)
            # getting one token from the distribution
            idx_next = torch.multinomial(props, num_samples=1)  # (B, 1)
            # append the new token to the input
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
