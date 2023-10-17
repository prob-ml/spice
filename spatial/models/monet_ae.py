"""A graph convolutional autoencoder for MERFISH data."""
from copy import deepcopy
import pytorch_lightning as pl
import torch
from torch_geometric.utils import degree

from spatial.models import base_networks


class BasicAEMixin(pl.LightningModule):

    """
    A method dump for models to be used under the Pytorch Lightning framework.
    Mixin implementing

    - loss calculations
    - training_step, validation_step,test_step,configure_optimizers for pytorchlightning
    """

    celltype_lookup = {
        "Ambiguous": 0,
        "Astrocyte": 1,
        "Endothelial 1": 2,
        "Endothelial 2": 3,
        "Endothelial 3": 4,
        "Ependymal": 5,
        "Excitatory": 6,
        "Inhibitory": 7,
        "Microglia": 8,
        "OD Immature 1": 9,
        "OD Immature 2": 10,
        "OD Mature 1": 11,
        "OD Mature 2": 12,
        "OD Mature 3": 13,
        "OD Mature 4": 14,
        "Pericytes": 15,
    }

    # The above description is important because these methods ONLY
    # get used in the child class where class variables are defined.
    def __init__(
        self,
        observables_dimension=155,
        hidden_dimensions=(512, 512, 512),
        latent_dimension=155,
        loss_type="mae",
        other_logged_losses=("mse"),
        radius=0,
        mask_random_prop=0,
        mask_cells_prop=0.05,
        mask_genes_prop=0,
        response_genes=None,
        celltypes=None,
        batchnorm=True,
        final_relu=False,
        attach_mask=False,
        dropout=0,
        responses=False,
        hide_responses=True,
        include_skip_connections=False,
        optimizer="sgd",
        aggr="mean",
        pseudo_mode="distance",
    ):
        super().__init__()
        self.observables_dimension = observables_dimension
        self.hidden_dimensions = hidden_dimensions
        self.latent_dimension = latent_dimension
        self.loss_type = loss_type
        self.other_logged_losses = other_logged_losses
        self.radius = radius
        self.mask_random_prop = mask_random_prop
        self.mask_cells_prop = mask_cells_prop
        self.mask_genes_prop = mask_genes_prop
        # needed so that during testing a different set
        # of responses other than MERFISH is useable.
        self.response_genes = response_genes
        self.celltypes = celltypes
        self.batchnorm = batchnorm
        self.final_relu = final_relu
        self.attach_mask = attach_mask
        self.dropout = dropout
        self.responses = responses
        self.hide_responses = hide_responses
        self.include_skip_connections = include_skip_connections
        self.optimizer = optimizer
        self.aggregation = aggr
        self.pseudo_mode = pseudo_mode

    @staticmethod
    def calc_loss(pred, val, losstype, celltype_data=None, celltype=None):
        # standard losses
        if losstype == "mse_against_log1pdata":
            return torch.mean((pred - torch.log(1 + val)) ** 2)
        elif losstype == "mse":
            return torch.mean((pred - val) ** 2)
        elif losstype == "mae":
            return torch.mean(torch.abs(pred - val))
        else:
            raise NotImplementedError

    def calc_pseudo(self, edge_index, pos):
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

        if self.pseudo_mode == "distance":
            coord1 = pos[edge_index[0]]
            coord2 = pos[edge_index[1]]
            edge_dir = coord2 - coord1
            rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
            # theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
            return rho  # torch.cat((rho), dim=1)

        if self.pseudo_mode == "polar":
            coord1 = pos[edge_index[0]]
            coord2 = pos[edge_index[1]]
            edge_dir = coord2 - coord1
            rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
            rho = rho if self.radius == 0 else rho / self.radius
            theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
            return torch.cat((rho, theta), dim=1)

        elif self.pseudo_mode == "degree":
            degrees = degree(edge_index[0])
            coord1 = (1 / torch.sqrt(degrees[edge_index[0]])).unsqueeze(-1)
            coord2 = (1 / torch.sqrt(degrees[edge_index[1]])).unsqueeze(-1)
            return torch.cat((coord1, coord2), dim=1)

        else:
            raise ValueError("Mode improperly or not specified.")

    def mask_cells(self, batch):
        n_cells = batch.x.shape[0]
        # 0 in masked_indeces means mask applied, 1 means not
        new_batch_obj = deepcopy(batch)
        masked_indeces = ~(
            torch.rand((n_cells, 1), device=new_batch_obj.x.device)
            < self.mask_cells_prop
        )
        new_batch_obj.x *= masked_indeces
        return new_batch_obj, masked_indeces

    def mask_genes(self, batch):
        n_genes = batch.x.shape[1]
        new_batch_obj = deepcopy(batch)
        # 0 in masked_indeces means mask applied, 1 means not
        masked_indeces = ~(
            torch.rand((1, n_genes), device=new_batch_obj.x.device)
            < self.mask_genes_prop
        )
        if self.responses:
            masked_indeces = torch.ones(
                (1, n_genes), device=new_batch_obj.x.device, dtype=bool
            )
            masked_indeces[:, self.response_genes] = ~(
                torch.rand((1, len(self.response_genes)), device=new_batch_obj.x.device)
                < self.mask_genes_prop
            )
        new_batch_obj.x *= masked_indeces

        return new_batch_obj, masked_indeces

    def mask_at_random(self, batch):

        # 0 in masked_indeces means mask applied, 1 means not
        n_cells, n_genes = batch.x.shape[0], batch.x.shape[1]
        masked_indeces = ~(
            torch.rand((n_cells, n_genes), device=batch.x.device)
            < self.mask_random_prop
        )
        if self.responses:
            masked_indeces = torch.ones(
                (n_cells, n_genes), device=batch.x.device, dtype=bool
            )
            masked_indeces[:, self.response_genes] = ~(
                torch.rand(n_cells, len(self.response_genes), device=batch.x.device)
                < self.mask_random_prop
            )
        new_batch_obj = deepcopy(batch)
        new_batch_obj.x *= masked_indeces

        return new_batch_obj, masked_indeces

    def mask_celltypes(self, batch, celltypes):
        # 0 in masked_indeces means mask applied, 1 means not
        if celltypes is None:
            return batch, torch.ones((batch.x.shape[0], 1), device=batch.x.device)
        celltype_values = [self.celltype_lookup[celltype] for celltype in celltypes]
        masked_indeces = ~(
            sum(batch.y[:, 1] == val for val in celltype_values).bool().reshape(-1, 1)
        )
        new_batch_obj = deepcopy(batch)
        masked_indeces = masked_indeces.to(new_batch_obj.x.device, non_blocking=True)
        new_batch_obj.x *= masked_indeces
        return new_batch_obj, masked_indeces

    def remove_responses(self, batch):
        new_batch_obj = deepcopy(batch)
        responses = torch.tensor(
            [0 if x in self.response_genes else 1 for x in range(batch.x.shape[1])],
            device=new_batch_obj.x.device,
        )
        new_batch_obj.x *= torch.tensor(responses)
        return new_batch_obj

    def training_step(self, batch, batch_idx):

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
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
        self.log("train_loss_" + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss_" + additional_loss,
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

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
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
                "val_loss_" + additional_loss,
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
    labelinfo = torch.tensor([])

    def test_step(self, batch, batch_idx):

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
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
                "test_loss_" + additional_loss,
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
        self.labelinfo = torch.cat((self.labelinfo, batch.y.cpu()), 0)

        return loss

    def configure_optimizers(self, scheduler=True):
        if self.optimizer["name"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer["params"])
        elif self.optimizer["name"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer["params"])
        if scheduler:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "min",
                    # MESS AROUND WITH THIS VALUE
                    # threshold=1e-5,
                    # threshold_mode="abs"
                    patience=3,
                ),
                "monitor": "val_loss",
            }
            return ([optimizer], [scheduler])
        return optimizer


class MonetDense(BasicAEMixin):
    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        output_dimension,
        loss_type,
        other_logged_losses,
        dim,
        kernel_size,
        radius,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        response_genes,
        celltypes,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
        hide_responses,
        include_skip_connections,
        optimizer,
        aggr,
        pseudo_mode,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        """
        # constuctor ensures that output dimension matches the observables dimension
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            output_dimension,
            loss_type,
            other_logged_losses,
            radius,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltypes,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
            hide_responses,
            include_skip_connections,
            optimizer,
            aggr,
            pseudo_mode,
        )

        self.dim = dim
        self.kernel_size = kernel_size
        self.output_dimension = output_dimension

        self.dense_network = base_networks.DenseReluGMMConvNetwork(
            [self.observables_dimension]
            + list(self.hidden_dimensions)
            + [self.output_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=self.include_skip_connections,
            aggr=self.aggregation,
        )

    def training_step(self, batch, batch_idx):

        _, responses = self(batch)
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("train_loss_" + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
                    additional_loss,
                ),
                prog_bar=True,
            )
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        _, responses = self(batch)
        # print(f"This validation batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("val_loss", loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "val_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
                    additional_loss,
                ),
                prog_bar=True,
            )
        return loss

    def test_step(self, batch, batch_idx):

        _, responses = self(batch)
        # print(f"This test batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        for additional_loss in self.other_logged_losses:
            self.log(
                "test_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
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
            responses,
            dataformats="HW",
        )

        self.inputs = torch.cat((self.inputs, batch.x.cpu()), 0)
        self.gene_expressions = torch.cat((self.gene_expressions, responses.cpu()), 0)
        self.labelinfo = torch.cat((self.labelinfo, batch.y.cpu()), 0)

        return loss

    def forward(self, batch):
        pseudo = self.calc_pseudo(batch.edge_index, batch.pos)
        non_response_genes = torch.ones(batch.x.shape[1], dtype=bool)
        non_response_genes[self.response_genes] = False
        output = self.dense_network(
            batch.x[:, non_response_genes], batch.edge_index, pseudo, num_gpus=3
        )
        return batch.x, output


class TrivialDense(BasicAEMixin):
    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        output_dimension,
        loss_type,
        other_logged_losses,
        radius,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        response_genes,
        celltypes,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
        hide_responses,
        include_skip_connections,
        optimizer,
    ):
        """
        observables_dimension -- number of values associated with each graph node
        hidden_dimensions -- list of hidden values to associate with each graph node
        output_dimension -- the output dimension of the dense network
        """
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            output_dimension,
            loss_type,
            other_logged_losses,
            radius,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltypes,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
            hide_responses,
            include_skip_connections,
            optimizer,
        )

        self.dense_network = base_networks.construct_dense_relu_network(
            [observables_dimension] + list(hidden_dimensions) + [output_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
        )

    def training_step(self, batch, batch_idx):

        _, responses = self(batch)
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("train_loss_" + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
                    additional_loss,
                ),
                prog_bar=True,
            )
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        _, responses = self(batch)
        # print(f"This validation batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("val_loss", loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "val_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
                    additional_loss,
                ),
                prog_bar=True,
            )
        return loss

    def test_step(self, batch, batch_idx):

        _, responses = self(batch)
        # print(f"This test batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            responses,
            batch.x[:, self.response_genes],
            self.loss_type,
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        for additional_loss in self.other_logged_losses:
            self.log(
                "test_loss_" + additional_loss,
                self.calc_loss(
                    responses,
                    batch.x[:, self.response_genes],
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
            responses,
            dataformats="HW",
        )

        self.inputs = torch.cat((self.inputs, batch.x.cpu()), 0)
        self.gene_expressions = torch.cat((self.gene_expressions, responses.cpu()), 0)
        self.labelinfo = torch.cat((self.labelinfo, batch.y.cpu()), 0)

        return loss

    def forward(self, batch):
        non_response_genes = torch.ones(batch.x.shape[1], dtype=bool)
        non_response_genes[self.response_genes] = False
        output = self.dense_network(batch.x[:, non_response_genes])
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
        radius,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        response_genes,
        celltypes,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
        hide_responses,
        include_skip_connections,
        optimizer,
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
            radius,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltypes,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
            hide_responses,
            include_skip_connections,
            optimizer,
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
        radius,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        response_genes,
        celltypes,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
        hide_responses,
        include_skip_connections,
        optimizer,
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
            radius,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltypes,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
            hide_responses,
            include_skip_connections,
            optimizer,
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
            include_skip_connections=self.include_skip_connections,
        )
        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [self.latent_dimension]
            + list(reversed(self.hidden_dimensions))
            + [self.observables_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=self.include_skip_connections,
        )

    def forward(self, batch):
        pseudo = self.calc_pseudo(batch.edge_index, batch.pos)
        latent_loadings = self.encoder_network(
            batch.x, batch.edge_index, pseudo, num_gpus=1
        )
        expr_reconstruction = self.decoder_network(
            latent_loadings, batch.edge_index, pseudo, num_gpus=1
        )
        return latent_loadings, expr_reconstruction


class MonetVAE(MonetAutoencoder2D):
    """Variational Autoencoder for graph data whose nodes are embedded in 2d"""

    def __init__(
        self,
        observables_dimension,
        hidden_dimensions,
        latent_dimension,
        loss_type,
        other_logged_losses,
        dim,
        kernel_size,
        radius,
        mask_random_prop,
        mask_cells_prop,
        mask_genes_prop,
        response_genes,
        celltypes,
        batchnorm,
        final_relu,
        attach_mask,
        dropout,
        responses,
        hide_responses,
        include_skip_connections,
        optimizer,
    ):
        super().__init__(
            observables_dimension,
            hidden_dimensions,
            latent_dimension,
            loss_type,
            other_logged_losses,
            dim,
            kernel_size,
            radius,
            mask_random_prop,
            mask_cells_prop,
            mask_genes_prop,
            response_genes,
            celltypes,
            batchnorm,
            final_relu,
            attach_mask,
            dropout,
            responses,
            hide_responses,
            include_skip_connections,
            optimizer,
        )

        self.decoder_network = base_networks.DenseReluGMMConvNetwork(
            [self.latent_dimension // 2]
            + list(reversed(self.hidden_dimensions))
            + [self.observables_dimension],
            use_batchnorm=self.batchnorm,
            dropout=self.dropout,
            dim=self.dim,
            kernel_size=self.kernel_size,
            include_skip_connections=self.include_skip_connections,
        )

    def training_step(self, batch, batch_idx):

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        z_means, z_logvars, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            z_means,
            z_logvars
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("train_loss_" + self.loss_type, loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "train_loss_" + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                    z_means,
                    z_logvars,
                ),
                prog_bar=True,
            )
        self.log("gpu_allocated", torch.cuda.memory_allocated() / (1e9), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        z_means, z_logvars, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            z_means,
            z_logvars
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        self.log("val_loss", loss, prog_bar=True)
        for additional_loss in self.other_logged_losses:
            self.log(
                "val_loss_" + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                    z_means,
                    z_logvars,
                ),
                prog_bar=True,
            )
        return loss

    gene_expressions = torch.tensor([])
    inputs = torch.tensor([])
    labelinfo = torch.tensor([])

    def test_step(self, batch, batch_idx):

        new_batch_obj, celltype_mask = self.mask_celltypes(batch, self.celltypes)
        new_batch_obj, random_mask = self.mask_at_random(new_batch_obj)
        new_batch_obj, gene_mask = self.mask_genes(new_batch_obj)
        new_batch_obj, cell_mask = self.mask_cells(new_batch_obj)
        if self.hide_responses:
            new_batch_obj = self.remove_responses(new_batch_obj)
        masking_tensor = (celltype_mask * random_mask * gene_mask * cell_mask).bool()
        if self.attach_mask:
            new_batch_obj.x = torch.cat((new_batch_obj.x, masking_tensor), dim=1)
        z_means, z_logvars, reconstruction = self(new_batch_obj)
        # print(f"This training batch has {batch.x.shape[0]} cells.")
        loss = self.calc_loss(
            reconstruction[~masking_tensor],
            batch.x[~masking_tensor],
            self.loss_type,
            z_means,
            z_logvars
            # celltype_data=batch.y[:, 1],
            # celltype="Excitatory",
        )
        for additional_loss in self.other_logged_losses:
            self.log(
                "test_loss_" + additional_loss,
                self.calc_loss(
                    reconstruction[~masking_tensor],
                    batch.x[~masking_tensor],
                    additional_loss,
                    z_means,
                    z_logvars,
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
        self.labelinfo = torch.cat((self.labelinfo, batch.y.cpu()), 0)

        return loss

    def calc_loss(self, pred, val, losstype, z_mean, z_logvar):
        reconstruction_loss = BasicAEMixin.calc_loss(pred, val, losstype)
        kl_loss = 0.5 * torch.mean(
            torch.sum(-(1 + z_logvar) + z_mean.pow(2) + z_logvar.exp(), 1)
        )
        return (
            reconstruction_loss + kl_loss
        )  # maximizing evidence - KL is the same as minimizing reconstruction + KL

    def reparameterize(self, z_mean, z_logvar):
        epsilon = torch.distributions.normal.Normal(0, 1).rsample()
        return z_mean + torch.exp(z_logvar) * epsilon

    def forward(self, batch):
        pseudo = self.calc_pseudo(batch.edge_index, batch.pos)
        latent_loadings = self.encoder_network(
            batch.x, batch.edge_index, pseudo, num_gpus=1
        )
        latent_dim = latent_loadings.shape[1]
        z_means, z_logvars = (
            latent_loadings[:, : (latent_dim // 2)],
            latent_loadings[:, (latent_dim // 2) :],
        )
        z_sample = self.reparameterize(z_means, z_logvars)
        expr_reconstruction = self.decoder_network(
            z_sample, batch.edge_index, pseudo, num_gpus=1
        )
        return z_means, z_logvars, expr_reconstruction
