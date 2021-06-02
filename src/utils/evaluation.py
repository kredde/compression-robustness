import torch
from pytorch_lightning.metrics.classification import Accuracy
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import pandas as pd

from hydra.utils import instantiate

from src.utils.config_utils import format_result
from src.datamodules.data.corruptions import SingleCurruptionDataloader


def test_model(model: LightningDataModule, criterion: torch.nn.Module, data_loader: DataLoader):
    """
        Test loop. Evaluates model on a given dataloader
        TODO: Add more metrics
    """
    model.eval()
    cpu = torch.device("cpu")

    accuracy = Accuracy()
    running_loss = 0
    predictions = np.ndarray((0, 10))
    targets = np.array([])

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.to(cpu)
            target = target.to(cpu)
            logits = model(image)
            loss = criterion(logits, target)
            preds = torch.argmax(logits, dim=1)
            predictions = np.vstack((predictions, logits.numpy()))
            targets = np.concatenate((targets, target.numpy()))

            acc = accuracy(preds, target)

            running_loss += loss * image.size(0)
    acc = accuracy.compute().item()
    loss = (running_loss / len(data_loader.dataset)).item()

    pred_df = pd.DataFrame(predictions)
    pred_df['targets'] = targets

    return [{'test/acc': acc, 'test/loss': loss}], pred_df


def test(model: LightningModule, datamodule: LightningDataModule,
         logger: LightningLoggerBase, name: str = None,
         config: DictConfig = None, path: str = None):
    """
        Test pipeline. Executes the test config for a given model
    """

    test_dataloader = datamodule.test_dataloader()
    test_result, predictions = test_model(model, model.criterion, test_dataloader)
    logger.log_metrics(format_result(test_result, name))
    predictions.to_csv(f"{path}/{f'{name}_' if name else ''}preds.csv", index=False)

    if config.get('test'):
        if config.test.get('corruptions'):
            c_config = config.test.corruptions

            sum_acc = 0
            for s in c_config.severities:
                for c in c_config.corruptions:
                    dataset = SingleCurruptionDataloader(
                        datamodule.data_c_test,
                        c,
                        s,
                        c_config.folder_name,
                        transform=datamodule.test_transforms,
                        dataset_path=c_config.dataset_path)

                    dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=datamodule.batch_size,
                        num_workers=datamodule.num_workers,
                        pin_memory=datamodule.pin_memory,
                        shuffle=False,
                    )

                    c_result, c_predictions = test_model(model, model.criterion, dataloader)
                    sum_acc += c_result[0]['test/acc']
                    logger.log_metrics(format_result(c_result, f'{name}_{c}_{str(s)}' if name else f'{c}_{str(s)}'))
                    c_predictions.to_csv(f"{path}/{f'{name}_' if name else ''}{c}_{str(s)}_preds.csv", index=False)

            acc = sum_acc / (len(c_config.corruptions) * len(c_config.severities))
            logger.log_metrics({f'{name}_c_test/acc' if name else 'c_test/acc': acc})

        if config.test.get('ood'):
            ood_datamodule = instantiate(config.ood)
            ood_datamodule.prepare_data()
            ood_datamodule.setup()
            ood_test = ood_datamodule.test_dataloader()

            ood_result, ood_predictions = test_model(model, model.criterion, ood_test)
            logger.log_metrics(format_result(ood_result, f'{name}_ood' if name else 'ood'))
            ood_predictions.to_csv(f"{path}/{f'{name}_' if name else ''}ood_preds.csv", index=False)

    # TODO: return metrics
    return None
