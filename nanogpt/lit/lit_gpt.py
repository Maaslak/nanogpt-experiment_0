from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import lightning
from hydra.utils import instantiate

from nanogpt.config import Conf
from nanogpt.models.gpt import NanoGPT

class LitNanoGPT(lightning.LightningModule):
    def __init__(self, config: Conf):
        self.config = config
        self.nano_gpt = NanoGPT(
            vocab_size=config.vocab_size,
            embedding_dim=config.gpt_conf.embedding_dim,
            block_size=config.block_size,
            n_heads=config.gpt_conf.n_heads,
            n_blocks=config.gpt_conf.n_blocks
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = instantiate(self.config.optim_partial)(self.nano_gpt.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, target = batch
        B, T = x.shape
        y_hat = self.nano_gpt(x)
        loss = torch.nn.functional.cross_entropy(
            y_hat.view(B * T, -1),
            target.reshape(B * T)
        )
        # add WB logging ? 
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, target = batch
        B, T = x.shape
        y_hat = self.nano_gpt(x)
        loss = torch.nn.functional.cross_entropy(
            y_hat.view(B * T, -1),
            target.reshape(B * T)
        )
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, max_compl=50) -> Any:
        compl = torch.concat([
            batch,
            torch.zeros((*batch.shape[:-1], max_compl), dtype=torch.long)],
            dim=1
        )
        last_id = batch.shape[-1]
        for i in range(last_id + 1, max_compl + last_id):
            start = max(last_id, i - self.config.block_size)
            logits, _ = self(compl[...,  start:i], targets=None)
            proba = torch.nn.functional.softmax(logits[..., i - start - 1, :], dim=-1)
            compl[..., i] = torch.multinomial(proba, 1).view(-1)
        
        return compl