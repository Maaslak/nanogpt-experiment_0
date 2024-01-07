import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
import torch
from hydra.utils import instantiate

from nanogpt.config import Conf, register_configs
from nanogpt.lit.lit_gpt import LitNanoGPT

logger = logging.getLogger(__name__)
register_configs()

def train_step(model: torch.nn.Module, optimizer: Optimizer, x: torch.Tensor, targets: torch.Tensor):
  model.train()
  _, loss = model(x, targets)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def train(cfg: DictConfig) -> None:
    logger.info("Config %s", cfg)
    cfg_obj: Conf = OmegaConf.to_object(cfg)
    model = LitNanoGPT(cfg_obj)
    dataset = 
    pass

if __name__ == "__main__":
    train()
