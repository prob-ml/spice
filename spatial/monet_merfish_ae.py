"""
Jeff Regier Lab
MIT License

A graph convolutional autoencoder.
"""

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.nn import (GMMConv, avg_pool, global_mean_pool, graclus,
                                max_pool, max_pool_x)
from torch_geometric.utils import degree

from merfish_geometric import Merfish


def calc_pseudo(edge_index, pos):
    """
    Input:
    - edge_index, an (N_edges x 2) long tensor indicating edges of a graph
    - pos, an (N_vertices x 2) float tensor indicating coordinates of nodes

    Output:
    - pseudo, an (N_edges x 2) float tensor indicating edge-values (to be used in graph-convnet)
    """
    coord1 = pos[edge_index[0]]
    coord2 = pos[edge_index[1]]
    edge_dir = coord2 - coord1
    rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
    theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
    return torch.cat((rho, theta), dim=1)


class MonetAutoencoder(pl.LightningModule):
    """
    Autoencoder for graph data whose nodes are embedded in 2d
    """

    def __init__(self, observables_dimension, latent_dimension):
        """
        observables_dimension -- number of values associated with each node of the graph
        latent_dimension -- number of latent values to associate with each node of the graph
        """
        super().__init__()

        self.conv1enc = GMMConv(observables_dimension, 50, dim=2, kernel_size=25)
        self.conv2enc = GMMConv(50, 25, dim=2, kernel_size=25)
        self.conv3enc = GMMConv(25, latent_dimension, dim=2, kernel_size=25)

        self.conv1dec = GMMConv(latent_dimension, 25, dim=2, kernel_size=25)
        self.conv2dec = GMMConv(25, 50, dim=2, kernel_size=25)
        self.conv3dec = GMMConv(50, observables_dimension, dim=2, kernel_size=25)

        self.batchnorm1enc = torch.nn.BatchNorm1d(50)
        self.batchnorm2enc = torch.nn.BatchNorm1d(25)
        self.batchnorm3enc = torch.nn.BatchNorm1d(latent_dimension)

        self.batchnorm1dec = torch.nn.BatchNorm1d(25)
        self.batchnorm2dec = torch.nn.BatchNorm1d(50)

        self.fc1 = nn.Linear(latent_dimension, latent_dimension)

    def encoder(self, X, E, pseudo):
        tmp = F.relu(self.batchnorm1enc(self.conv1enc(X, E, pseudo)))
        tmp = F.relu(self.batchnorm2enc(self.conv2enc(tmp, E, pseudo)))
        Z = self.fc1(F.relu(self.batchnorm3enc(self.conv3enc(tmp, E, pseudo))))
        return Z

    def decoder(self, Z, E, pseudo):
        tmp = F.relu(self.batchnorm1dec(self.conv1dec(Z, E, pseudo)))
        tmp = F.relu(self.batchnorm2dec(self.conv2dec(tmp, E, pseudo)))
        reconstruction = self.conv3dec(tmp, E, pseudo)
        return reconstruction

    def forward(self, batch):
        # observable data at each vertex
        X = batch.x.type("torch.cuda.FloatTensor")
        E = batch.edge_index.type("torch.cuda.LongTensor")
        P = batch.pos.type("torch.cuda.FloatTensor")

        # calculate edge weights
        pseudo = calc_pseudo(E, P)

        # run encoder
        Z = self.encoder(X, E, pseudo)

        # run decoder
        X_reconstruction = self.decoder(Z, E, pseudo)

        return X_reconstruction

    def training_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = F.poisson_nll_loss(reconstruction, batch.x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = F.poisson_nll_loss(reconstruction, batch.x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


if __name__ == "__main__":
    train75_loader = Merfish("../data", train=True)
    merfish_train, mnist_val = random_split(train75_loader, [139, 13])
    merfish_test = Merfish("../data", train=False)

    model = MonetAutoencoder(159, 2)
    trainer = pl.Trainer(
        flush_logs_every_n_steps=12, log_every_n_steps=12, gpus=1, max_epochs=100
    )
    trainer.fit(
        model,
        DataLoader(merfish_train, batch_size=1, num_workers=2),
        DataLoader(mnist_val, batch_size=1, shuffle=False),
    )
    trainer.test(model, DataLoader(merfish_test, batch_size=1))
