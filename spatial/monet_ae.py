"""
A graph convolutional autoencoder for MERFISH data.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv


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


class TrivialAutoencoder(pl.LightningModule):
    """
    Autoencoder for graph data, ignoring the graph structure
    """

    def __init__(self, observables_dimension, latent_dimension, loss_type):
        """
        observables_dimension -- number of values associated with each node of the graph
        latent_dimension -- number of latent values to associate with each node of the graph
        """
        super().__init__()

        self.loss_type = loss_type

        self.conv1enc = torch.nn.Linear(observables_dimension, 50)
        self.conv2enc = torch.nn.Linear(50, 25)
        self.conv3enc = torch.nn.Linear(25, latent_dimension)

        self.conv1dec = torch.nn.Linear(latent_dimension, 25)
        self.conv2dec = torch.nn.Linear(25, 50)
        self.conv3dec = torch.nn.Linear(50, observables_dimension)

        self.batchnorm1enc = torch.nn.BatchNorm1d(50)
        self.batchnorm2enc = torch.nn.BatchNorm1d(25)
        self.batchnorm3enc = torch.nn.BatchNorm1d(latent_dimension)

        self.batchnorm1dec = torch.nn.BatchNorm1d(25)
        self.batchnorm2dec = torch.nn.BatchNorm1d(50)

        self.fc1 = nn.Linear(latent_dimension, latent_dimension)

    def encoder(self, expr, edges, pseudo):
        tmp = F.relu(self.batchnorm1enc(self.conv1enc(expr)))
        tmp = F.relu(self.batchnorm2enc(self.conv2enc(tmp)))
        latent = self.fc1(F.relu(self.batchnorm3enc(self.conv3enc(tmp))))
        return latent

    def calc_loss(self, pred, val):
        if self.loss_type == "mse_against_log1pdata":
            return torch.sum((pred - torch.log(1 + val)) ** 2)
        elif self.loss_type == "mse":
            return torch.sum((pred - val) ** 2)
        else:
            raise NotImplementedError(self.loss_type)

    def decoder(self, latent, edges, pseudo):
        tmp = F.relu(self.batchnorm1dec(self.conv1dec(latent)))
        tmp = F.relu(self.batchnorm2dec(self.conv2dec(tmp)))
        reconstruction = self.conv3dec(tmp)
        return reconstruction

    def forward(self, batch):
        # calculate edge weights
        pseudo = calc_pseudo(batch.edge_index, batch.pos)

        # run encoder
        latent_loadings = self.encoder(batch.x, batch.edge_index, pseudo)

        # run decoder
        expr_reconstruction = self.decoder(latent_loadings, batch.edge_index, pseudo)

        return expr_reconstruction

    def training_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
        # return torch.optim.SGD(self.parameters(), lr=0.001)


class MonetAutoencoder(pl.LightningModule):
    """
    Autoencoder for graph data whose nodes are embedded in 2d
    """

    def __init__(self, observables_dimension, latent_dimension, loss_type):
        """
        observables_dimension -- number of values associated with each node of the graph
        latent_dimension -- number of latent values to associate with each node of the graph
        """
        super().__init__()

        self.loss_type = loss_type

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

    def encoder(self, expr, edges, pseudo):
        tmp = F.relu(self.batchnorm1enc(self.conv1enc(expr, edges, pseudo)))
        tmp = F.relu(self.batchnorm2enc(self.conv2enc(tmp, edges, pseudo)))
        latent = self.fc1(F.relu(self.batchnorm3enc(self.conv3enc(tmp, edges, pseudo))))
        return latent

    def calc_loss(self, pred, val):
        if self.loss_type == "mse_against_log1pdata":
            return torch.sum((pred - torch.log(1 + val)) ** 2)
        elif self.loss_type == "mse":
            return torch.sum((pred - val) ** 2)
        else:
            raise NotImplementedError(self.loss_type)

    def decoder(self, latent, edges, pseudo):
        tmp = F.relu(self.batchnorm1dec(self.conv1dec(latent, edges, pseudo)))
        tmp = F.relu(self.batchnorm2dec(self.conv2dec(tmp, edges, pseudo)))
        reconstruction = self.conv3dec(tmp, edges, pseudo)
        return reconstruction

    def forward(self, batch):
        # calculate edge weights
        pseudo = calc_pseudo(batch.edge_index, batch.pos)

        # run encoder
        latent_loadings = self.encoder(batch.x, batch.edge_index, pseudo)

        # run decoder
        expr_reconstruction = self.decoder(latent_loadings, batch.edge_index, pseudo)

        return expr_reconstruction

    def training_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = batch.to("cuda")
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=0.001)
        return torch.optim.Adam(self.parameters())
