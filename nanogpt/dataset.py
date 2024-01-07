import lightning

from nanogpt.config import Conf

class LitWiki2(lightning.LightningDataModule):
    def __init__(self, config: Conf):
        super().__init__()
        self.config = config
    
    def setup(self, stage: str):
        self.tokenizer