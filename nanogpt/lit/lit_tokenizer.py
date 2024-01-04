import lightning

from nanogpt.config import ModelConf

class Tokenizer(lightning.LightningModule):
    def __init__(self, config: ModelConf):
        self.config = config

    