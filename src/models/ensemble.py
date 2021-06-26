import torch
from torch import nn
import torch.nn.functional as F
import abc
from src.models.base import BaseModel


class BaseEnsemble(BaseModel):
    def __init__(self, models=[], lr: float = 0.1, weight_decay: float = 5e-4, num_classes: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.requires_fit = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.models = nn.ModuleList(models)
        self.criterion = torch.nn.CrossEntropyLoss()

    def __len__(self):
        """
        Return the number of models in the ensemble.
        """
        return len(self.models)

    def __getitem__(self, index):
        """Return the `index`-th model in the ensemble."""
        return self.models[index]

    @abc.abstractmethod
    def forward(self, x):
        """
            Implementation of forward pass
        """

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay)

    def fuse_model(self):
        for model in self.models:
            model.fuse_model()


class BaggingEnsemble(BaseEnsemble):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
            Average over class distributions from all models
        """
        outputs = [
            model(x) for model in self.models
        ]
        proba = sum(outputs) / len(outputs)

        return proba


class StackingEnsemble(BaseEnsemble):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_fit = True
        self.fc = nn.Linear(len(self) * self.num_classes, self.num_classes)

    def forward(self, x):
        """
            Concatenate outputs and predict using meta learning fc layer
        """

        with torch.no_grad():
            outputs = torch.cat([
                model(x) for model in self.models
            ], dim=1)

        logits = self.fc(outputs)

        return logits
