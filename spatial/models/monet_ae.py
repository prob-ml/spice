"""A graph convolutional autoencoder for MERFISH data."""

import pytorch_lightning as pl
import torch

from spatial.models import base_networks


def calc_pseudo(edge_index, pos):
    """
    Calculate pseudo

    Input:
      - edge_index, an (N_edges x 2) long tensor indicating edges of a graph
      - pos, an (N_vertices x 2) float tensor indicating coordinates of nodes

    Output:
      - pseudo, an (N_edges x 2) float tensor indicating edge-values
        (to be used in graph-convnet)
    """
    coord1 = pos[edge_index[0]]
    coord2 = pos[edge_index[1]]
    edge_dir = coord2 - coord1
    rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
    theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
    return torch.cat((rho, theta), dim=1)


class BasicAEMixin(pl.LightningModule):
    """
    Mixin implementing

    - loss calculations
    - training_step, validation_step,test_step,configure_optimizers for pytorchlightning
    """

    def calc_loss(self, pred, val):
        if self.loss_type == "mse_against_log1pdata":
            return torch.sum((pred - torch.log(1 + val)) ** 2)
        elif self.loss_type == "mse":
            return torch.sum((pred - val) ** 2)
        else:
            raise NotImplementedError(self.loss_type)

    def training_step(self, batch, batch_idx):
        _, reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class TrivialAutoencoder(BasicAEMixin):
    """Autoencoder for graph data, ignoring the graph structurea"""

    def __init__(
        self, observables_dimension, hidden_dimensions, latent_dimension, loss_type
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type

        self.encoder_network = base_networks.construct_dense_relu_network(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
        )

        self.decoder_network = base_networks.construct_dense_relu_network(
            [latent_dimension]
            + list(reversed(hidden_dimensions))
            + [observables_dimension],
        )

    def forward(self, batch):

        latent_loadings = self.encoder_network(batch.x)
        expr_reconstruction = self.decoder_network(latent_loadings)
        return latent_loadings, expr_reconstruction


class MonetAutoencoder2D(BasicAEMixin):
    """Autoencoder for graph data whose nodes are embedded in 2d"""

    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        latent_dimension,
        loss_type,
        dim,
        kernel_size,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type

        self.encoder_network = base_networks.DenseReluGMMConvNetwork(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
            dim=2,
            kernel_size=25,
        )
        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [latent_dimension]
            + list(reversed(hidden_dimensions))
            + [observables_dimension],
            dim=2,
            kernel_size=25,
        )

    def forward(self, batch):
        pseudo = calc_pseudo(batch.edge_index, batch.pos)
        latent_loadings = self.encoder_network(batch.x, batch.edge_index, pseudo)
        expr_reconstruction = self.decoder_network(
            latent_loadings, batch.edge_index, pseudo
        )
        return latent_loadings, expr_reconstruction
