from argparse import ArgumentParser

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader

from colored_mnist_data import ColoredMNISTDataModule, SimpleDataset

import numpy as np

seed = 1239754


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default=".")

    return parser


class DMLEstimateX(LightningModule):
    def __init__(self, input_dim: int, lr: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.input_dim = input_dim

        self.sur_layers = nn.ModuleList([nn.Sequential(nn.Linear(input_dim - 1, 1),
                                                       nn.Sigmoid()) for i in range(input_dim)])

    def _forward(self, x):
        return torch.cat([layer(torch.cat([x[:, 0:i], x[:, (i+1):]], dim=1)) for i, layer in enumerate(self.sur_layers)],
                         dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self._forward(x)

    def step(self, batch):
        x, y = batch
        x = x.view(x.shape[0], -1)
        xp = self._forward(x)
        loss = nn.functional.mse_loss(xp, x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='train_loss', value=loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='val_loss', value=loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DMLEstimateY(LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, lr: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.sur_layers = nn.ModuleList([nn.Linear(input_dim - 1, 1) for i in range(input_dim)])

    def _forward(self, x):
        return torch.cat([layer(torch.cat([x[:, 0:i], x[:, (i+1):]], dim=1)) for i, layer in enumerate(self.sur_layers)],
                         dim=-1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self._forward(x)

    def step(self, batch):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self._forward(x).mean(-1).view(-1, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='train_loss', value=loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='val_loss', value=loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DMLEstimateXY(LightningModule):
    def __init__(self, input_dim: int, lr: float, dml: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.is_dml = dml

        self.layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layer(x).mean(dim=-1)

    def step(self, batch):
        x, y = batch
        x = x.view(x.shape[0], -1)
        logits = self.layer(x)
        if not self.is_dml:
            logits = logits.mean(dim=-1).view(-1, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='train_loss', value=loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log(name='val_loss', value=loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_second(args=None):
    seed_everything(seed)
    np.random.seed(seed)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--DML", dest='dml', action='store_true')
    parser.add_argument("--X_dir", type=str, default=".")
    parser.add_argument("--Y_dir", type=str, default=".")

    args = parser.parse_args(args)
    args.random_seed = seed

    dm = ColoredMNISTDataModule.from_argparse_args(args)
    dm.setup()
    train_dl = dm.train_dataloader()
    true_val_dl = dm.val_dataloader()

    # def get_dataloader(dl):
    #     X = dl.dataset.X.view(dl.dataset.X.shape[0], -1)
    #     model_x = DMLEstimateX.load_from_checkpoint(args.X_dir)
    #     model_x.eval()
    #     X_res = X - model_x(dl.dataset.X)
    #
    #     y = dl.dataset.y.view(dl.dataset.y.shape[0], -1)
    #     model_y = DMLEstimateY.load_from_checkpoint(args.Y_dir)
    #     model_y.eval()
    #     y_res = y - model_y(X)
    #
    #     data = SimpleDataset(X_res, y_res)
    #     torch.save(data, './DML_dataset.pt')
    #     data = torch.load('./DML_dataset.pt')
    #     return DataLoader(data, batch_size=dm.batch_size)
    #
    # args.input_dim = np.prod(list(dm.size()))
    # if not args.dml:
    #     checkpoint_path = './linearXY'
    # else:
    #     train_dl = get_dataloader(train_dl)
    #     val_dl = get_dataloader(true_val_dl)
    #     checkpoint_path = './DML'

    # model = DMLEstimateXY(**vars(args))
    def _mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    # model = DMLEstimateXY.load_from_checkpoint('./DML/lightning_logs/version_9/checkpoints/epoch=0-step=195.ckpt')
    model = DMLEstimateXY.load_from_checkpoint('./linearXY/lightning_logs/version_3/checkpoints/epoch=9-step=489.ckpt')
    model.eval()
    logits = model(true_val_dl.dataset.X).view(-1, 1)
    print(nn.functional.binary_cross_entropy_with_logits(logits, true_val_dl.dataset.y))
    print(_mean_accuracy(logits, true_val_dl.dataset.y))
    # args.default_root_dir = checkpoint_path

    # trainer = Trainer.from_argparse_args(args)
    # trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)
    # return dm, model, trainer
    # print validation error
    # trainer.validate(ckpt_path='best', dataloaders=true_val_dl, verbose=True)


def train_first(args=None):
    seed_everything(seed)
    np.random.seed(seed)

    parser = ArgumentParser()

    parser = add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--X", dest='X', action='store_true')
    args = parser.parse_args(args)
    args.random_seed = seed

    dm = ColoredMNISTDataModule.from_argparse_args(args)
    args.input_dim = np.prod(list(dm.size()))
    if args.X:
        model = DMLEstimateX(**vars(args))
        checkpoint_path = './X'
    else:
        model = DMLEstimateY(**vars(args))
        checkpoint_path = './Y'
    args.default_root_dir = checkpoint_path

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer


# mnist_train = DataLoader(mnist_train, batch_size=64)
if __name__ == '__main__':
    # train_first()
    train_second()