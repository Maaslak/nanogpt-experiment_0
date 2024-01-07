import torch

from torch import nn

from nanogpt.models.nn import Block, FeadForward


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