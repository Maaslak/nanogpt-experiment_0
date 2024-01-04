from typing import Any
from functools import partial
from dataclasses import dataclass, field, asdict

from torch.optim import AdamW

@dataclass
class GPTConf:
  embedding_dim: int = 24
  n_heads: int = 4
  n_blocks: int = 1

@dataclass
class TrainConf:
  max_iter: int = 100_000
  log_every: int = 5_000

@dataclass
class ModelConf:
  vocab_size: int = 500
  block_size: int = 10
  batch_size: int = 8
  optim_partial: Any = partial(AdamW, lr=10e-4)
  force_tokenizer_retrain: bool = False
  min_len: int = 100
  gpt_conf: GPTConf = field(default_factory=GPTConf)
  train_conf: TrainConf = field(default_factory=TrainConf)
  tokenizer_template: str =  "utils/tokenizer/tokenizer_vocab_{vocab_size}.json"
  
  @property
  def tokenizer_path(self):
    return self.tokenizer_template.format(vocab_size=self.vocab_size)