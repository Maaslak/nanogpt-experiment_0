import hydra

from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

from nanogpt.config import ModelConf

cs = ConfigStore.instance()
cs.store(name="base_config", node=ModelConf)

@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config"
)
def train(cfg: ModelConf) -> None:
    pass

if __name__ == "__main__":
    train()
