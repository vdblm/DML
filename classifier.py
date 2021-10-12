from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torch import nn, optim, autograd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from colored_mnist_data import ColoredMNISTDataModule

from pytorch_lightning import LightningModule, Trainer, seed_everything

from pytorch_lightning import loggers as pl_loggers

tb_logger = pl_loggers.TensorBoardLogger("logs/")


class Classifier(LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, lr: float, **kwargs):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        lin_layers = [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1)]
        for lin in lin_layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.main = nn.Sequential(lin_layers[0], nn.ReLU(True), lin_layers[1], nn.ReLU(True), lin_layers[2])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.main(x)

    @staticmethod
    def _mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    @staticmethod
    def _mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def step(self, batch):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.main(x)
        loss = self._mean_nll(logits, y)
        acc = self._mean_accuracy(logits, y)

        logs = {
            "accuracy": acc,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        # calculate loss + log
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--input_dim", type=int, default=784)
        parser.add_argument("--lr", type=float, default=1e-3)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


def train(args=None):
    seed = 1239754
    seed_everything(seed)
    np.random.seed(seed)

    parser = ArgumentParser()

    parser = Classifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)
    args.random_seed = seed

    dm = ColoredMNISTDataModule.from_argparse_args(args)
    args.input_dim = np.prod(list(dm.size()))
    model = Classifier(**vars(args))

    args.logger = tb_logger
    args.callbacks = [EarlyStopping(monitor="val_accuracy")]
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer


if __name__ == '__main__':
    dm, model, trainer = train()

