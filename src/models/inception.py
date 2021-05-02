from torchvision.models.quantization.inception import inception_v3
import torch
from torch.quantization import fuse_modules

from src.models.base import BaseModel 


class InceptionV3(BaseModel):
   def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, *args, **kwargs):
      super().__init__(*args, **kwargs)

      self.model = inception_v3()

      # this line ensures params passed to LightningModule will be saved to ckpt
      self.save_hyperparameters()

   def configure_optimizers(self):
      return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
   def fuse_model(self):
     self.model.fuse_model()

               