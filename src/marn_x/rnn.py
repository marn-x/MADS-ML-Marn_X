import torch
from torch import nn

Tensor = torch.Tensor

import mlflow
from datetime import datetime
from dataclasses import dataclass

from pathlib import Path

from mltrainer import TrainerSettings, ReportTypes
from mltrainer.metrics import Accuracy, Metric

from mltrainer import rnn_models, Trainer
from torch import optim

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import PaddedPreprocessor
preprocessor = PaddedPreprocessor()

gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
streamers = gesturesdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]

trainstreamer = train.stream()
validstreamer = valid.stream()

accuracy = Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()

class GeneralizationGap(Metric):
    """
    Computes the absolute difference between training and validation accuracy.
    Inputs must be scalar tensors or floats representing each accuracy.
    """

    def _compute(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        if not isinstance(train_acc, torch.Tensor):
            train_acc = torch.tensor(train_acc)
        if not isinstance(val_acc, torch.Tensor):
            val_acc = torch.tensor(val_acc)
        return torch.abs(train_acc - val_acc)

    def __repr__(self) -> str:
        return "Generalization Gap (|train_acc - val_acc|)"

import torch
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

settings = TrainerSettings(
    epochs=10,
    metrics=[accuracy],
    logdir=Path("gestures"),
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TOML, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
    optimizer_kwargs= {'lr': 1e-3, 'weight_decay': 1e-05},
    scheduler_kwargs={"factor": 0.1, "patience": 30},
    earlystop_kwargs= None
)

class GRUmodel(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

class LSTMmodel(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.LSTM(
            config.input_size,
            config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gestures")
modeldir = Path("gestures").resolve()
if not modeldir.exists():
    modeldir.mkdir(parents=True)

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    horizon: int
    output_size: int
    dropout: float = 0.0

    def __iter__(self):
        """Allow iteration over all attributes."""
        return iter(vars(self))
    
    def __getitem__(self, key):
        """Allow dictionary-style access to attributes."""
        return vars(self)[key]

    def __len__(self) -> int:
        """Return the number of attributes."""
        return len(vars(self))
    
    def items(self):
        return vars(self).items()


with mlflow.start_run():
    mlflow.set_tag("dev", "marn_x")
    config = ModelConfig(
        input_size=3,
        hidden_size=32,
        num_layers=2,
        horizon=20,
        output_size=20,
        dropout=0.3
    )
    mlflow.log_params(config)

    mlflow.set_tag("model", "LSTM")
    # model = GRUmodel(config=config)
    model = LSTMmodel(config=config)

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=trainstreamer,
        validdataloader=validstreamer,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device=device,
    )
    trainer.loop()

    tag = datetime.now().strftime("%Y%m%d-%H%M")
    modelpath = modeldir / (tag + "model.pt")
    torch.save(model, modelpath)