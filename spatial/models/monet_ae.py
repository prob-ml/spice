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
    def __init__(
        self,
        observables_dimension=155,
        hidden_dimensions=(512, 512, 512),
        latent_dimension=155,
        loss_type="mae",
        other_logged_losses=("mse"),
        mask_random_prop=0,
        mask_cells_prop=0.05,
        mask_genes_prop=0,
        response_genes=None,
        celltype_lookup=0,
        batchnorm=True,
        final_relu=False,
        attach_mask=False,
        dropout=0,
        responses=False,
    ):
        super().__init__()
        self.observables_dimension = observables_dimension
        self.hidden_dimensions = hidden_dimensions
        self.latent_dimension = latent_dimension
        self.loss_type = loss_type
        self.other_logged_losses = other_logged_losses
        self.mask_random_prop = mask_random_prop
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop
        # needed so that during testing a different set
        # of responses other than MERFISH is useable.
        self.response_genes = response_genes
        self.celltype_lookup = celltype_lookup
        self.batchnorm = batchnorm
        self.final_relu = final_relu
        self.attach_mask = attach_mask
        self.dropout = dropout
        self.responses = responses

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
                    pred[:, torch.tensor(self.response_genes)]
                    - val[:, torch.tensor(self.response_genes)]
                )
            )
        elif losstype == "mse_response":
            return torch.mean(
                (
                    pred[:, torch.tensor(self.response_genes)]
                    - val[:, torch.tensor(self.response_genes)]
                )
                ** 2
            )
        elif losstype == "mae_response_celltype":
            return torch.mean(
                torch.abs(
                    pred[:, torch.tensor(self.response_genes)]
                    - val[:, torch.tensor(self.response_genes)]
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
        # 0 in masked_indeces means mask applied, 1 means not
        masked_indeces = ~(torch.rand((n_cells, 1)) < self.mask_cells_prop)
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.to(new_batch_obj.x.device)
        new_batch_obj.x *= masked_indeces
        return new_batch_obj, masked_indeces

    def mask_genes(self, batch):
        n_genes = batch.x.shape[1]
        # 0 in masked_indeces means mask applied, 1 means not
        masked_indeces = ~(torch.rand((1, n_genes)) < self.mask_genes_prop)
        if self.responses:
            masked_indeces = torch.ones((1, n_genes), dtype=bool)
            masked_indeces[:, self.response_genes] = ~(
                torch.rand(1, len(self.response_genes)) < self.mask_genes_prop
            )
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.to(new_batch_obj.x.device)
        new_batch_obj.x *= masked_indeces

        return new_batch_obj, masked_indeces

    def mask_at_random(self, batch):

        # 0 in masked_indeces means mask applied, 1 means not
        n_cells, n_genes = batch.x.shape[0], batch.x.shape[1]
        masked_indeces = ~(torch.rand((n_cells, n_genes)) < self.mask_random_prop)
        if self.responses:
            masked_indeces = torch.ones((n_cells, n_genes), dtype=bool)
            masked_indeces[:, self.response_genes] = ~(
                torch.rand(n_cells, len(self.response_genes)) < self.mask_random_prop
            )
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.to(new_batch_obj.x.device)
        new_batch_obj.x *= masked_indeces

        return new_batch_obj, masked_indeces

    def training_step(self, batch, batch_idx):

        new_batch_obj, random_mask = self.mask_at_random(batch)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        masking_tensor = random_mask * gene_mask * cell_mask
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        _, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("train_loss: " + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss: " + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                ),
                prog_bar=True,
            )
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        new_batch_obj, random_mask = self.mask_at_random(batch)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        masking_tensor = random_mask * gene_mask * cell_mask
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        _, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("val_loss", loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "val_loss: " + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                ),
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
        masking_tensor = random_mask * gene_mask * cell_mask
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        _, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        for additional_loss in self.other_logged_losses:
            self.log(
                "test_loss: " + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                ),
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


class MonetDense(BasicAEMixin):
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
        response_genes,
        celltype_lookup,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        """
        # constuctor ensures that output dimension matches the observables dimension
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            latent_dimension,
            loss_type,
            other_logged_losses,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltype_lookup,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
        )

        self.dim = dim
        self.kernel_size = kernel_size

        self.dense_network = base_networks.DenseReluGMMConvNetwork(
            [self.observables_dimension]
            + list(self.hidden_dimensions)
            + [self.latent_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=False,
        )

    def forward(self, batch):
        pseudo = calc_pseudo(batch.edge_index, batch.pos)
        output = self.dense_network(batch.x, batch.edge_index, pseudo)
        return batch.x, output


class TrivialAutoencoder(BasicAEMixin):
    """Autoencoder for graph data, ignoring the graph structure"""

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
        response_genes,
        celltype_lookup,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            latent_dimension,
            loss_type,
            other_logged_losses,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltype_lookup,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
        )

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
        response_genes,
        celltype_lookup,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        latent_dimension -- number of latent values to associate with each graph node
        """
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            latent_dimension,
            loss_type,
            other_logged_losses,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltype_lookup,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
        )

        self.dim = dim
        self.kernel_size = kernel_size

        self.encoder_network = base_networks.DenseReluGMMConvNetwork(
            [self.observables_dimension]
            + list(self.hidden_dimensions)
            + [self.latent_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=False,
        )
        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [self.latent_dimension]
            + list(reversed(self.hidden_dimensions))
            + [self.observables_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=False,
        )

    def forward(self, batch):
        pseudo = calc_pseudo(batch.edge_index, batch.pos)
        latent_loadings = self.encoder_network(batch.x, batch.edge_index, pseudo)
        expr_reconstruction = self.decoder_network(
            latent_loadings, batch.edge_index, pseudo
        )
        return latent_loadings, expr_reconstruction
