import lightning

from nanogpt.config import ModelConf

class LitWiki2(lightning.LightningDataModule):
    def __init__(self, config: ModelConf):
        super().__init__()
        self.config = config
    
    def setup(self, stage: str):
