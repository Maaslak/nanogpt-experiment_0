from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import lightning

from nanogpt.config import ModelConf
from nanogpt.nn.gpt import NanoGPT

class LitNanoGPT(lightning.LightningModule):
    def __init__(self, config: ModelConf, device: torch.device):
        self.config = config
        self.model = NanoGPT(
            vocab_size=config.vocab_size,
            embedding_dim=config.gpt_conf.embedding_dim,
            block_size=config.block_size,
            n_heads=config.gpt_conf.n_heads,
            n_blocks=config.gpt_conf.n_blocks
        ).to(device)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.config.optim_partial(self.model.parameters())
