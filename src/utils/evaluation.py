import torch
from pytorch_lightning.metrics.classification import Accuracy
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase
from tqdm import tqdm
from omegaconf import DictConfig

from src.utils.config_utils import format_result
from src.datamodules.data.corruptions import SingleCurruptionDataloader


def test_model(model: LightningDataModule, criterion: torch.nn.Module, data_loader: DataLoader) -> Dict:
    """
        Test loop. Evaluates model on a given dataloader
        TODO: Add more metrics
    """
    model.eval()
    cpu = torch.device("cpu")

    accuracy = Accuracy()
    running_loss = 0

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = image.to(cpu)
            target = target.to(cpu)
            logits = model(image)
            loss = criterion(logits, target)
            preds = torch.argmax(logits, dim=1)

            acc = accuracy(preds, target)

            running_loss += loss * image.size(0)
    acc = accuracy.compute().item()
    loss = (running_loss / len(data_loader.dataset)).item()

    return [{'test/acc': acc, 'test/loss': loss}]


def test(model: LightningModule, datamodule: LightningDataModule,
         logger: LightningLoggerBase, name: str = None,
         config: DictConfig = None):
    """
        Test pipeline. Executes the test config for a given model
    """

    test_dataloader = datamodule.test_dataloader()
    test_result = test_model(model, model.criterion, test_dataloader)
    logger.log_metrics(format_result(test_result, name))

    if config.get('test'):
        c_config = config.test.corruptions

        sum_acc = 0
        for s in c_config.severities:
            for c in c_config.corruptions:
                dataset = SingleCurruptionDataloader(
                    datamodule.data_c_test,
                    c,
                    s,
                    c_config.folder_name,
                    transform=datamodule.test_transforms)

                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=datamodule.batch_size,
                    num_workers=datamodule.num_workers,
                    pin_memory=datamodule.pin_memory,
                    shuffle=False,
                )

                c_result = test_model(model, model.criterion, dataloader)
                sum_acc += c_result[0]['test/acc']
                logger.log_metrics(format_result(c_result, f'{name}_{c}_{str(s)}' if name else f'{c}_{str(s)}'))

        acc = sum_acc / (len(c_config.corruptions) * len(c_config.severities))
        logger.log_metrics({f'{name}_c_test/acc' if name else 'c_test/acc': acc})

    # TODO: return metrics
    return None
