"""A graph convolutional autoencoder for MERFISH data."""

import pytorch_lightning as pl
import torch

from spatial.models import base_networks

# from torch._C import device


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

    def calc_loss(self, pred, val, alpha=0.5):
        if self.loss_type == "mse_against_log1pdata":
            return torch.sum((pred - torch.log(1 + val)) ** 2)
        elif self.loss_type == "mse":
            return torch.mean((pred - val) ** 2)
        elif self.loss_type == "mae":
            return torch.mean((pred - val).abs())
        elif self.loss_type == "convex_loss":
            return alpha * torch.mean((pred - val) ** 2) + (1 - alpha) * torch.mean(
                (pred - val).abs()
            )
        else:
            raise NotImplementedError(self.loss_type)

    def mask_cells(self, batch):
        n_cells = batch.x.shape[0]
        masked_indeces = torch.rand((n_cells, 1)) < self.mask_cells_prop
        new_batch_obj = batch.x * masked_indeces
        return new_batch_obj

    def mask_genes(self, batch):
        n_genes = batch.x.shape[1]
        masked_indeces = torch.rand((1, n_genes)) < self.mask_genes_prop
        new_batch_obj = batch.x * masked_indeces
        return new_batch_obj

    def training_step(self, batch, batch_idx):
        if self.mask_cells_prop > 0:
            _, reconstruction = self(self.mask_cells(batch))
        else:
            _, reconstruction = self(batch)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("train_loss", loss, prog_bar=True)
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    gene_expressions = torch.tensor([])
    inputs = torch.tensor([])

    def test_step(self, batch, batch_idx):
        _, reconstruction = self(batch)
        loss = self.calc_loss(reconstruction, batch.x)
        self.log("test_loss", loss, prog_bar=True)

        # save input and output images
        tensorboard = self.logger.experiment
        tensorboard.add_image(
            f"{self.__class__.__name__}/{self.logger.version}/{batch_idx}/input",
            batch.x,
            dataformats="HW",
        )
        tensorboard.add_image(
            f"{self.__class__.__name__}/"
            f"{self.logger.version}/{batch_idx}/reconstruction",
            reconstruction,
            dataformats="HW",
        )

        self.inputs = torch.cat((self.inputs, batch.x.cpu()), 0)
        self.gene_expressions = torch.cat(
            (self.gene_expressions, reconstruction.cpu()), 0
        )

        return loss

    # def test_step_end(self, output_results):
    #     # this out is now the full size of the batch
    #     self.L1_losses.append(
    #         torch.nn.functional.l1_loss(self.inputs, self.gene_expressions)
    #     )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class TrivialAutoencoder(BasicAEMixin):
    """Autoencoder for graph data, ignoring the graph structurea"""

    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        latent_dimension,
        loss_type,
        mask_cells_prop,
        mask_genes_prop,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop

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
        mask_cells_prop,
        mask_genes_prop,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop

        self.encoder_network = base_networks.DenseReluGMMConvNetwork(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
            dim=dim,
            kernel_size=kernel_size,
        )
        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [latent_dimension]
            + list(reversed(hidden_dimensions))
            + [observables_dimension],
            dim=dim,
            kernel_size=kernel_size,
        )

    def forward(self, batch):
        pseudo = calc_pseudo(batch.edge_index, batch.pos)
        latent_loadings = self.encoder_network(batch.x, batch.edge_index, pseudo)
        expr_reconstruction = self.decoder_network(
            latent_loadings, batch.edge_index, pseudo
        )
        return latent_loadings, expr_reconstruction
