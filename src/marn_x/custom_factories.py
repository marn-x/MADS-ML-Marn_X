from typing import cast

from mads_datasets.settings import ImgDatasetSettings, FileTypes
from mads_datasets.factories.torchfactories import ImgDataset
from mads_datasets.base import AbstractDatasetFactory, DatasetProtocol
from mads_datasets.datatools import iter_valid_paths
from pydantic import HttpUrl
from pathlib import Path

import torch
from torchvision import transforms

import random
from pathlib import Path
from typing import (
    Any,
    Mapping,
)

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

eurosat_data_transforms = transforms.Compose([
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

class AugmentPreprocessor():
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X = [self.transform(x) for x in X]
        return torch.stack(X), torch.stack(y)