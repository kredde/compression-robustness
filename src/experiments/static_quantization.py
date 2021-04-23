from typing import Optional
from pytorch_lightning import LightningModule
import torch
from torch.utils.data.dataloader import DataLoader

from src.utils.quantization_util import calibrate_model


def quantize_static(model: LightningModule, dataloader: DataLoader, backend: Optional[str] = None, num_calibration_batches: int = 32, *args, **kwargs):
    """
      Performs static post training quantization on a given model
      https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization

      args:
        model: the model do be quantized, needs to implement `fuse_model`
        dataloader: the training dataloader used to calibrate the activation functions
        backend
        num_calibration_batches: the number of batches used to calibrate the model
    """
    model.eval()

    # fuse layers for quantization
    model.fuse_model()

    # specify quantization config
    # observers are inserted to determine how the different activations should be quantized at inference time
    if not backend:
        model.qconfig = torch.quantization.default_qconfig
    else:
        # TODO: enable passing custom configs like # of bits
        model.qconfig = torch.quantization.get_default_qconfig(backend)
    print('QConfig ', backend)
    print(model.qconfig)

    model = torch.quantization.prepare(model, inplace=False)

    # Calibrate with the training set
    calibrate_model(model, model.criterion, dataloader,
                    neval_batches=num_calibration_batches)

    # convert model
    torch.quantization.convert(model, inplace=True)

    return model
