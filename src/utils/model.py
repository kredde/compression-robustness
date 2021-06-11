from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
import os


def load_experiment(path: str, checkpoint: str = 'last.ckpt', compressed_path: str = None):
    """
      Loads an existing model and its dataloader.

      args:
        path (string): The path to the log folder
        checkpoint (string): the name of the checkpoint file
    """
    # load conf
    config: DictConfig = OmegaConf.load(path + '/.hydra/config.yaml')

    # reinitialize model and datamodule
    model = get_class(config.model._target_)
    datamodule: LightningDataModule = instantiate(config.datamodule)
    datamodule.setup()

    if "seed" in config:
        seed_everything(config.seed)

    if compressed_path:
        path = compressed_path
    else:
        # find best checkpoint
        if checkpoint == 'best':
            files = os.listdir(path + '/checkpoints')
            for f in files:
                if f.startswith('epoch='):
                    checkpoint = f
        path += '/checkpoints/' + checkpoint

    model = model.load_from_checkpoint(path)

    return model, datamodule, config
