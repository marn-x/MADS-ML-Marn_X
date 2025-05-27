from typing import cast

from mads_datasets.settings import ImgDatasetSettings, FileTypes
from mads_datasets.factories.torchfactories import ImgDataset
from mads_datasets.base import AbstractDatasetFactory, DatasetProtocol
from mads_datasets.datatools import iter_valid_paths
from pydantic import HttpUrl
from pathlib import Path

from torch import nn

from torchvision import transforms
import torch

import hashlib
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
from loguru import logger

from mads_datasets.datatools import create_headers, get_file
from mads_datasets.settings import DatasetSettings, SecureDatasetSettings

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"
    logger.warning("This model will take 15-20 minutes on CPU. Consider using accelaration, eg with google colab (see button on top of the page)")
logger.info(f"Using {device}")

eurosatsettings = ImgDatasetSettings(
    dataset_url=cast(
        HttpUrl,
        "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
    ),
    filename=Path("EuroSAT_RGB.zip"),
    name="EuroSAT_RGB",
    unzip=True,
    formats=[FileTypes.JPG],
    trainfrac=0.8,
    img_size=(64, 64),
    digest="c8fa014336c82ac7804f0398fcb19387",
)

data_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class EurosatDatasetFactory(AbstractDatasetFactory[ImgDatasetSettings]):
    def create_dataset(
        self, *args: Any, **kwargs: Any
    ) -> Mapping[str, DatasetProtocol]:
        self.download_data()
        formats = self._settings.formats
        paths_, class_names = iter_valid_paths(
            self.subfolder / "2750", formats=formats 
        )
        paths = [*paths_]
        random.shuffle(paths)
        trainidx = int(len(paths) * self._settings.trainfrac)
        train = paths[:trainidx]
        valid = paths[trainidx:]
        traindataset = ImgDataset(train, class_names, img_size=self._settings.img_size)
        validdataset = ImgDataset(valid, class_names, img_size=self._settings.img_size)
        return {"train": traindataset, "valid": validdataset}

eurosatfactory = EurosatDatasetFactory(eurosatsettings, datadir=Path.home() / ".cache/mads_datasets")

class AugmentPreprocessor():
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)

streamers = eurosatfactory.create_datastreamer(batchsize=32)

trainprocessor= AugmentPreprocessor(data_transforms)
# validprocessor = AugmentPreprocessor(data_transforms["val"])

train = streamers["train"]
valid = streamers["valid"]
train.preprocessor = trainprocessor
valid.preprocessor = trainprocessor
trainstreamer = train.stream()
validstreamer = valid.stream()

print(next(iter(trainstreamer)))

import torchvision
from torchvision.models import resnet18, ResNet18_Weights
resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)


from mltrainer import metrics
accuracy = metrics.Accuracy()

from torch import optim
optimizer = optim.SGD
scheduler = optim.lr_scheduler.StepLR

from mltrainer import Trainer, TrainerSettings, ReportTypes

settings = TrainerSettings(
    epochs=30,
    metrics=[accuracy],
    logdir="modellogs/eurosat",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.MLFLOW],
    optimizer_kwargs= {'lr': 0.1, 'weight_decay': 1e-05, 'momentum': 0.9},
    scheduler_kwargs= {'step_size' : 10, 'gamma' : 0.1},
    earlystop_kwargs= None,
)

trainer = Trainer(
    model=resnet,
    settings=settings,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=scheduler,
    device=device,
)

trainer.loop()