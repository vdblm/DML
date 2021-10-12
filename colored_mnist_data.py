from typing import Optional

import numpy as np
import torch

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


class ColoredMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32, random_seed: int = 39575):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (14, 14, 2)

        self.mnist_train, self.mnist_val = None, None
        self.seed = random_seed

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist = MNIST(self.data_dir, train=True, download=False)
            mnist_train = (mnist.data[:50000], mnist.targets[:50000])
            mnist_val = (mnist.data[50000:], mnist.targets[50000:])

            rng_state = np.random.get_state()
            np.random.shuffle(mnist_train[0].numpy())
            np.random.set_state(rng_state)
            np.random.shuffle(mnist_train[1].numpy())

            train1 = self._make_colored_data(mnist_train[0][::2], mnist_train[1][::2], 0.25, 0.2)
            train2 = self._make_colored_data(mnist_train[0][1::2], mnist_train[1][1::2], 0.25, 0.1)
            val = self._make_colored_data(mnist_val[0], mnist_val[1], 0.25, 0.9)

            self.mnist_train = SimpleDataset(torch.cat([train1['images'], train2['images']], dim=0),
                                             torch.cat([train1['labels'], train2['labels']], dim=0))

            self.mnist_val = SimpleDataset(val['images'], val['labels'])
            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # TODO test dataset
        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        #
        #     # Optionally...
        #     # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    @staticmethod
    def _make_colored_data(mnist_data, mnist_labels, p, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        mnist_data = mnist_data.reshape((-1, 28, 28))[:, ::2, ::2] # 2x subsample
        # Assign a binary label based on the digit; flip label with probability `p`
        labels = (mnist_labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(p, len(labels)))
        # Assign a color based on the label; flip the color with probability `e`
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([mnist_data, mnist_data], dim=-1)
        images[torch.tensor(range(len(images))), :, :, (1 - colors).long()] *= 0

        return {
            'images': (images.float() / 255.),
            'labels': labels[:, None]
        }