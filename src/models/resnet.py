from torchvision.models.quantization.resnet import resnet18, resnet50
import torch
from torch.quantization import fuse_modules
from torch.optim.lr_scheduler import MultiStepLR

from src.models.base import BaseModel 


class ResNet18(BaseModel):
   """
      https://arxiv.org/pdf/1512.03385.pdf (Section 4.2)
   """
   def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
      super().__init__(*args, **kwargs)

      self.model = resnet18(num_classes=num_classes)

      # this line ensures params passed to LightningModule will be saved to ckpt
      self.save_hyperparameters()

   def configure_optimizers(self):
      optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95), 'name': 'expo_lr'}
      return [optimizer], [lr_scheduler]
        
   def fuse_model(self):
     self.model.fuse_model()

class ResNet50(BaseModel):
   def __init__(self, lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
      super().__init__(*args, **kwargs)

      self.model = resnet50(num_classes=num_classes)

      # this line ensures params passed to LightningModule will be saved to ckpt
      self.save_hyperparameters()

   def configure_optimizers(self):
      optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95), 'name': 'expo_lr'}

      return [optimizer], [lr_scheduler]
        
   def fuse_model(self):
     self.model.fuse_model()
               