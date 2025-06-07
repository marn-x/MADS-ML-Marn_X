from custom_factories import AugmentPreprocessor, EurosatDatasetFactory, eurosatsettings, eurosat_data_transforms
from pathlib import Path

if __name__ == "__main__":
    eurosatfactory = EurosatDatasetFactory(eurosatsettings, datadir=Path.home() / ".cache/mads_datasets")
    preprocessor = AugmentPreprocessor(eurosat_data_transforms)

    train = streamers["train"]
    valid = streamers["valid"]
    train.preprocessor = trainprocessor
    valid.preprocessor = trainprocessor
    trainstreamer = train.stream()
    validstreamer = valid.stream()