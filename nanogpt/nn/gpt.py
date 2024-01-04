import torch

from torch import nn


class BiGramModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, vocab_size)

  def forward(self, ids, targets=None):
    logits = self.embedding(ids)
    B, T, C = logits.shape
    loss = None
    if targets is not None:
      # print(targets)
      loss = nn.functional.cross_entropy(
          logits.view(B * T, C),
          targets.reshape(B * T)
      )
    return logits, loss
  


class SelfAttentionHead(nn.Module):
  def __init__(self, embedding_dim, head_size, block_size):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.value_proj = nn.Linear(embedding_dim, head_size, bias=False)
    self.key_proj = nn.Linear(embedding_dim, head_size, bias=False)
    self.query_proj = nn.Linear(embedding_dim, head_size, bias=False)

    self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

  def forward(self, x):
    B, T, E = x.shape

    query = self.query_proj(x)
    key = self.key_proj(x)
    value = self.value_proj(x)

    weights = (query @ key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.embedding_dim))

    weights = weights.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
    weights = torch.nn.functional.softmax(weights, dim=-1)
    # print(weights)
    return weights @ value

class MultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, n_heads):
    super().__init__()
    self.head_size = embedding_dim // n_heads
    if self.head_size * n_heads != embedding_dim:
      raise ValueError("n_heads has to be a factor of embedding_dim")
    self.heads = nn.ModuleList([
        SelfAttentionHead(embedding_dim, self.head_size)
        for _ in range(n_heads)
    ])
  def forward(self, x):
    return torch.cat([
        head(x)
        for head in self.heads
    ], axis=-1)

class FeadForward(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(embedding_dim, 2 * embedding_dim),
        nn.LayerNorm(2 * embedding_dim),
        nn.ReLU(),
        nn.Linear(2 * embedding_dim, embedding_dim),
        nn.LayerNorm(embedding_dim),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.layers(x)

class Block(nn.Module):
  def __init__(self, embedding_dim, n_heads):
    super().__init__()
    self.multiatt = MultiHeadAttention(embedding_dim, n_heads)
    self.ff = FeadForward(embedding_dim)
    self.ln = nn.LayerNorm(embedding_dim)

  def forward(self, x):
    x = x + self.multiatt(self.ln(x))
    x = x + self.ff(x)
    return x


class NanoGPT(nn.Module):
  def __init__(self, vocab_size, embedding_dim, block_size, n_heads, n_blocks):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim) # C, E
    self.position_embedding = nn.Embedding(block_size, embedding_dim) # T, E
    self.attention = nn.Sequential(
        *[Block(embedding_dim, n_heads)
        for _ in range(n_blocks)]
    )
    self.ff = FeadForward(embedding_dim)
    self.proj = nn.Linear(embedding_dim, vocab_size)

  def forward(self, ids, targets=None):
    x = self.embedding(ids)
    B, T, E = x.shape
    x = x + self.position_embedding(torch.arange(T))

    x = self.attention(x)
    x = self.ff(x)
    out = self.proj(x)

    loss = None
    if targets is not None:
      # print(targets)
      loss = torch.nn.functional.cross_entropy(
          out.view(B * T, -1),
          targets.reshape(B * T)
      )
    return out, loss