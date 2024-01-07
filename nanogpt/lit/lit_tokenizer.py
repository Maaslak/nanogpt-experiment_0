import lightning

from nanogpt.config import Conf

class Tokenizer(lightning.LightningModule):
    def __init__(self, config: Conf):
        self.config = config

    