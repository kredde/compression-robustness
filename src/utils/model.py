from pytorch_lightning import LightningModule, LightningDataModule, seed_everything
from hydra.utils import instantiate, get_class, get_original_cwd
from omegaconf import DictConfig, OmegaConf
import mlflow
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
    if config.get('model'):
        model = get_class(config.model._target_)
    elif config.get('ensemble'):
        model = get_class(config.ensemble._target_)
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


def load_exprerimant_by_id(exp_id: str, config: DictConfig):
    log_dir = None

    # instantiate client
    client = mlflow.tracking.MlflowClient(
        tracking_uri=config.logger.mlflow.tracking_uri)

    # get old experimant log dir using old conf
    data = client.get_run(exp_id).to_dictionary()
    log_dir: str = data['data']['params']['hydra/log_dir']

    # load the saved model and datamodule
    if not log_dir.startswith('/'):
        log_dir = get_original_cwd() + '/' + log_dir
    return load_experiment(
        log_dir, checkpoint="best", compressed_path=config.get('compressed_path'))
