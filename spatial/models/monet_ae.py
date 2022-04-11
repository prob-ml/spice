"""A graph convolutional autoencoder for MERFISH data."""
from copy import deepcopy
import pytorch_lightning as pl
import torch
from torch_geometric.utils import degree

from spatial.models import base_networks

# from torch._C import device


def calc_pseudo(edge_index, pos, mode="polar"):
    """
    Calculate pseudo

    Input:
      - edge_index, an (N_edges x 2) long tensor indicating edges of a graph
      - pos, an (N_vertices x 2) float tensor indicating coordinates of nodes
      - mode, "polar" or "degree" mode for pseudo-coordinate generation

    Output:
      - pseudo, an (N_edges x 2) float tensor indicating edge-values
        (to be used in graph-convnet)
    """

    if mode == "polar":
        coord1 = pos[edge_index[0]]
        coord2 = pos[edge_index[1]]
        edge_dir = coord2 - coord1
        rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
        # theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
        return rho  # torch.cat((rho), dim=1)

    elif mode == "degree":
        degrees = degree(edge_index[0])
        coord1 = (1 / torch.sqrt(degrees[edge_index[0]])).unsqueeze(-1)
        coord2 = (1 / torch.sqrt(degrees[edge_index[1]])).unsqueeze(-1)
        return torch.cat((coord1, coord2), dim=1)

    else:
        raise ValueError("Mode improperly or not specified.")


class BasicAEMixin(pl.LightningModule):

    """
    A method dump for models to be used under the Pytorch Lightning framework.
    Mixin implementing

    - loss calculations
    - training_step, validation_step,test_step,configure_optimizers for pytorchlightning
    """

    # The above description is important because these methods ONLY
    # get used in the child class where class variables are defined.
    saved_original_input = 0
    hopefully_masked_input = 0

    def calc_loss(self, pred, val, losstype, celltype_data=None, celltype=None):
        # standard losses
        if losstype == "mse_against_log1pdata":
            return torch.mean((pred - torch.log(1 + val)) ** 2)
        elif losstype == "mse":
            return torch.mean((pred - val) ** 2)
        elif losstype == "mae":
            return torch.mean(torch.abs(pred - val))

        # response losses
        elif losstype == "mae_response":
            return torch.mean(
                torch.abs(
                    pred[:, torch.tensor(self.responses)]
                    - val[:, torch.tensor(self.responses)]
                )
            )
        elif losstype == "mse_response":
            return torch.mean(
                (
                    pred[:, torch.tensor(self.responses)]
                    - val[:, torch.tensor(self.responses)]
                )
                ** 2
            )
        elif losstype == "mae_response_celltype":
            return torch.mean(
                torch.abs(
                    pred[:, torch.tensor(self.responses)]
                    - val[:, torch.tensor(self.responses)]
                )[
                    (celltype_data == self.celltype_lookup[celltype]).nonzero(
                        as_tuple=True
                    )[0]
                ]
            )

        else:
            raise NotImplementedError

    def mask_cells(self, batch):
        n_cells = batch.x.shape[0]
        masked_indeces = torch.rand((n_cells, 1)) < self.mask_cells_prop
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.type_as(new_batch_obj.x)
        new_batch_obj.x *= 1.0 - masked_indeces
        return new_batch_obj, masked_indeces

    def mask_genes(self, batch, responses=True):
        n_genes = batch.x.shape[1]
        masked_indeces = torch.rand((1, n_genes)) < self.mask_genes_prop
        if responses:
            masked_indeces = torch.zeros((1, n_genes), dtype=bool)
            masked_indeces[:, self.responses] = (
                torch.rand(1, len(self.responses)) < self.mask_genes_prop
            )
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.type_as(new_batch_obj.x)
        new_batch_obj.x *= 1.0 - masked_indeces

        return new_batch_obj, masked_indeces

    def mask_at_random(self, batch, responses=True):

        n_cells, n_genes = batch.x.shape[0], batch.x.shape[1]
        masked_indeces = torch.rand((n_cells, n_genes)) < self.mask_random_prop
        if responses:
            masked_indeces = torch.zeros((n_cells, n_genes), dtype=bool)
            masked_indeces[:, self.responses] = (
                torch.rand(n_cells, len(self.responses)) < self.mask_random_prop
            )
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.type_as(new_batch_obj.x)
        new_batch_obj.x *= 1.0 - masked_indeces

        return new_batch_obj, masked_indeces

    def training_step(self, batch, batch_idx):

        new_batch_obj, random_mask = self.mask_at_random(batch)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        _, reconstruction = self(new_batch_obj)
        masking_tensor = (1 - random_mask) * (1 - gene_mask) * (1 - cell_mask)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction * masking_tensor,
            batch.x * masking_tensor,
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("train_loss: " + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss: " + additional_loss,
                self.calc_loss(reconstruction, batch.x, additional_loss),
                prog_bar=True,
            )
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        new_batch_obj, random_mask = self.mask_at_random(batch)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        _, reconstruction = self(new_batch_obj)
        masking_tensor = (1 - random_mask) * (1 - gene_mask) * (1 - cell_mask)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction * masking_tensor,
            batch.x * masking_tensor,
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("val_loss", loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "val_loss: " + additional_loss,
                self.calc_loss(reconstruction, batch.x, additional_loss),
                prog_bar=True,
            )
        return loss

    gene_expressions = torch.tensor([])
    inputs = torch.tensor([])
    celltypes = torch.tensor([])

    def test_step(self, batch, batch_idx):
        new_batch_obj, random_mask = self.mask_at_random(batch)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        _, reconstruction = self(new_batch_obj)
        masking_tensor = (1 - random_mask) * (1 - gene_mask) * (1 - cell_mask)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction * masking_tensor,
            batch.x * masking_tensor,
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        for additional_loss in self.other_logged_losses:
            self.log(
                "test_loss: " + additional_loss,
                self.calc_loss(reconstruction, batch.x, additional_loss),
                prog_bar=True,
            )
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
        self.celltypes = torch.cat((self.celltypes, batch.y.cpu()), 0)

        return loss

    # def test_step_end(self, output_results):
    #     # this out is now the full size of the batch
    #     self.L1_losses.append(
    #         torch.nn.functional.l1_loss(self.inputs, self.gene_expressions)
    #     )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class BasicNN(BasicAEMixin):
    def training_step(self, batch, batch_idx):
        if self.mask_cells_prop > 0:
            mean_estimate = self(self.mask_cells(batch))
        else:
            mean_estimate = self(batch)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(mean_estimate, batch.x[:, self.responses], self.loss_type)
        self.log("train_loss", loss, prog_bar=True)
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mean_estimate = self(batch)
        loss = self.calc_loss(mean_estimate, batch.x[:, self.responses], self.loss_type)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    gene_expressions = torch.tensor([])
    inputs = torch.tensor([])
    celltypes = torch.tensor([])

    def test_step(self, batch, batch_idx):
        mean_estimate = self(batch)
        loss = self.calc_loss(mean_estimate, batch.x[:, self.responses], self.loss_type)
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
            mean_estimate,
            dataformats="HW",
        )

        self.inputs = torch.cat((self.inputs, batch.x[:, self.features].cpu()), 0)
        self.gene_expressions = torch.cat(
            (self.gene_expressions, mean_estimate.cpu()), 0
        )
        self.celltypes = torch.cat((self.celltypes, batch.y.cpu()), 0)

        return loss


# NN for learning single mean expression per gene.
class MeanExpressionNN(BasicNN):
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

        # no decoder needed as we are trying to stay in the "latent space"
        self.encoder_network = base_networks.construct_dense_relu_network(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
        )

    def forward(self, batch):

        latent_loadings = self.encoder_network(batch.x[:, self.features])
        return latent_loadings


class TrivialAutoencoder(BasicAEMixin):
    """Autoencoder for graph data, ignoring the graph structurea"""

    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        latent_dimension,
        loss_type,
        other_logged_losses,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        responses,
        celltype_lookup,
        batchnorm,
        final_relu,
        dropout,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type
        self.other_logged_losses = other_logged_losses
        self.mask_random_prop = mask_random_prop
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop
        # needed so that during testing a different set
        # of responses other than MERFISH is useable.
        self.responses = responses
        self.celltype_lookup = celltype_lookup
        self.batchnorm = batchnorm
        self.final_relu = final_relu
        self.dropout = dropout

        self.encoder_network = base_networks.construct_dense_relu_network(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
        )
        self.decoder_network = base_networks.construct_dense_relu_network(
            [latent_dimension]
            + list(reversed(hidden_dimensions))
            + [observables_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
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
        other_logged_losses,
        dim,
        kernel_size,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        responses,
        celltype_lookup,
        batchnorm,
        final_relu,
        dropout,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__()

        self.loss_type = loss_type
        self.other_logged_losses = other_logged_losses
        self.mask_random_prop = mask_random_prop
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop
        # needed so that during testing a different set
        # of responses other than MERFISH is useable.
        self.responses = responses
        self.celltype_lookup = celltype_lookup
        self.batchnorm = batchnorm
        self.final_relu = final_relu
        self.dropout = dropout

        self.encoder_network = base_networks.DenseReluGMMConvNetwork(
            [observables_dimension] + list(hidden_dimensions) + [latent_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=dim,
            kernel_size=kernel_size,
        )
        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [latent_dimension]
            + list(reversed(hidden_dimensions))
            + [observables_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
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
