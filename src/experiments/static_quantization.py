from pytorch_lightning import LightningModule
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from torch.utils.data.dataloader import DataLoader

from src.utils import quantization_util as utils


def quantize_static(model: LightningModule, dataloader: DataLoader, num_calibration_batches: int = 32,
                    activation_precision: int = 7, weight_precision: int = 8, *args, **kwargs):
    """
      Performs static post training quantization on a given model
      https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization

      args:
        model: the model do be quantized, needs to implement `fuse_model`
        dataloader: the training dataloader used to calibrate the activation functions
        backend
        num_calibration_batches: the number of batches used to calibrate the model
    """

    # x86 compatible quantization engine
    torch.backends.quantized.engine = 'fbgemm'

    # fuse layers for quantization
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    model.eval()

    assert 2 <= activation_precision and activation_precision <= 7
    assert 2 <= weight_precision and weight_precision <= 8

    activation_precision = utils.UINT_BOUNDS[activation_precision]
    weight_precision = utils.INT_BOUNDS[weight_precision]

    # specify quantization config
    # TODO: maybe use HistogramObserver?
    model.qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            quant_min=activation_precision[0],
            quant_max=activation_precision[1],
            qscheme=torch.per_tensor_affine
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            quant_min=weight_precision[0],
            quant_max=weight_precision[1],
            qscheme=torch.per_channel_affine
        )
    )

    print(model.qconfig)

    q_model = torch.quantization.prepare(model, inplace=False)

    # Calibrate with the training set
    utils.calibrate_model(q_model, q_model.criterion, dataloader,
                          neval_batches=num_calibration_batches)

    # convert model
    torch.quantization.convert(q_model, inplace=True)

    return q_model
