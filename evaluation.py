import logging
import dotenv
import hydra
from omegaconf import DictConfig
import torch


dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="eval_config.yaml")
def main(config: DictConfig):
    logger.info(config.pretty())

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import mlflow
    from hydra import utils
    from src.eval import eval
    from src.utils import model as model_utils, config_utils, ensemble_utils

    config_utils.extras(config)

    # get the hydra logdir using the exp_id
    if config.get('exp_id'):
        model, datamodule, exp_config = model_utils.load_exprerimant_by_id(config.exp_id, config)
    elif config.get('ensemble_models'):
        models = []
        datamodule = None
        exp_config = None
        for exp_id in config.ensemble_models:
            m, datamodule, exp_config = model_utils.load_exprerimant_by_id(exp_id, config)
            models.append(m)

        model = utils.instantiate(config.ensemble, models=models)

    else:
        # TODO: Find a easier way to load a past configuration
        raise Exception(
            '`exp_id` or `ensemble_models` must be defined in order to evaluate an existing experiment')

    # instanciate mlflow and the trainer for the evaluation
    mlf_logger = utils.instantiate(
        config.logger.mlflow, experiment_name=config.get('exp_name') or exp_config.logger.mlflow.experiment_name)
    trainer = utils.instantiate(
        config.trainer, callbacks=[], logger=[mlf_logger], _convert_='partial'
    )

    if hasattr(model, 'requires_fit') and model.requires_fit:
        ensemble_utils.fit_meta(model, trainer, datamodule)

    return eval(config, model, trainer, datamodule)


if __name__ == "__main__":
    main()
