import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dataclasses import asdict


from nanogpt.config import PydanticModelConf

logger = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def train(cfg: DictConfig) -> None:
    cfg = PydanticModelConf(OmegaConf.to_object(cfg))
    logger.info("Config %s", cfg.model_dump_json())
    pass

if __name__ == "__main__":
    train()
