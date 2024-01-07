from typing import Any, Callable, TypedDict
from functools import partial
from dataclasses import asdict, field

from pydantic.dataclasses import dataclass
from pydantic import field_validator, field_serializer, RootModel
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore

@dataclass
class GPTConf:
  embedding_dim: int = 24
  n_heads: int = 4
  n_blocks: int = 1

@dataclass
class TrainConf:
  max_iter: int = 100_000
  log_every: int = 5_000  

# class OptimPartial(TypedDict, total=False):
#   _target_: str = "torch.optim.AdamW"
#   lr: float = 1e-4

@dataclass
class ModelConf:
  vocab_size: int = 500
  block_size: int = 10
  batch_size: int = 8
  optim_partial: dict = field(default_factory=lambda: {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-4})
  force_tokenizer_retrain: bool = False
  min_len: int = 100
  gpt_conf: GPTConf = field(default_factory=GPTConf)
  train_conf: TrainConf = field(default_factory=TrainConf)
  tokenizer_template: str =  "utils/tokenizer/tokenizer_vocab_{vocab_size}.json"
  
  @property
  def tokenizer_path(self):
    return self.tokenizer_template.format(vocab_size=self.vocab_size)
  
  def __post_init__(self):
    self._optim_dict = self.optim_partial

  @field_validator("optim_partial", mode="after")
  @classmethod
  def to_partial(cls, v: dict) -> Callable:
    return instantiate(v)
  
  @field_serializer("optim_partial")
  def serialize_optim_partial(self, optim_partial: Callable, _info) -> dict:
    return self._optim_dict
  
def register_configs():
  cs = ConfigStore.instance()
  cs.store(name="base_config", node=ModelConf)

register_configs()
PydanticModelConf = RootModel[ModelConf]