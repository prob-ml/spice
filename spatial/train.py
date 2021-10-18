import os
import sys

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from spatial.merfish_dataset import MerfishDataset
from spatial.models import monet_ae

_models = [
    monet_ae.TrivialAutoencoder,
    monet_ae.MonetAutoencoder2D,
    monet_ae.MeanExpressionNN,
]
models = {cls.__name__: cls for cls in _models}

# specify logger (taken from bliss)
def setup_logger(cfg):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=cfg.paths.output,
            name=cfg.training.logger_name,
            version=(
                f"{cfg.model.name}__{cfg.model.kwargs.observables_dimension}"
                f"__{cfg.model.kwargs.hidden_dimensions}__"
                f"{cfg.model.kwargs.latent_dimension}__{cfg.n_neighbors}"
                f"__{cfg.model.kwargs.mask_cells_prop}"
            ),
        )
    return logger


# set up model saving (taken from bliss)
def setup_checkpoint_callback(cfg, logger):
    callbacks = []
    output = cfg.paths.output
    if cfg.training.trainer.checkpoint_callback:
        checkpoint_dir = f"{output}/lightning_logs/checkpoints/{cfg.model.name}"
        checkpoint_dir = os.path.join(output, checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            prefix="",
            filename=f"{cfg.model.name}__{cfg.model.kwargs.observables_dimension}"
            f"__{cfg.model.kwargs.hidden_dimensions}__"
            f"{cfg.model.kwargs.latent_dimension}__{cfg.n_neighbors}"
            f"__{cfg.model.kwargs.mask_cells_prop}",
        )
        callbacks.append(checkpoint_callback)

    return callbacks


def setup_early_stopping(cfg, callbacks):
    early_stop_callback = False
    if cfg.training.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
        )
        callbacks.append(early_stop_callback)
    return callbacks


def train(cfg: DictConfig, data=None):

    # setup training data
    if data is None:
        data = MerfishDataset(
            cfg.paths.data,
            train=True,
            log_transform=cfg.training.log_transform,
            non_response_genes_file=cfg.datasets.predictor_genes,
        )
    n_data = len(data)
    train_n = round(n_data * 11 / 12)
    train_data, val_data = random_split(data, [train_n, n_data - train_n])

    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size, num_workers=2
    )
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, num_workers=2)

    # ensuring data dimension is correct
    if cfg.model.kwargs.observables_dimension != data[0].x.shape[1]:
        raise AssertionError("Data dimension not in line with observables dimension.")

    # get response indeces so they can be passed into the model
    if data.responses is not None:
        OmegaConf.update(cfg, "model.kwargs.responses", data.responses)

    # setup logger
    logger = setup_logger(cfg)

    # setup checkpoints
    callbacks = setup_checkpoint_callback(cfg, logger)

    callbacks = setup_early_stopping(cfg, callbacks)

    # setup trainer
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(
        dict(
            logger=logger,
            callbacks=callbacks,
        )
    )
    trainer = pl.Trainer(**trainer_dict)

    # specify model
    model = models[cfg.model.name](**cfg.model.kwargs)

    # save model info
    if cfg.training.save_model_summary:

        output = cfg.paths.output

        # architecture
        try:
            sys.stdout = open(
                os.path.join(output, f"architecture/{cfg.model.name}.txt"), "w"
            )
            print(model)
            sys.stdout.close()
        except FileNotFoundError:
            os.makedirs(os.path.join(output, "architecture/"))
            sys.stdout = open(
                os.path.join(output, f"architecture/{cfg.model.name}.txt"), "w"
            )
            print(model)
            sys.stdout.close()

        # parameters (and model memory size)
        try:
            sys.stdout = open(
                os.path.join(output, f"parameters/{cfg.model.name}.txt"), "w"
            )
            print(model.summarize(mode=trainer.weights_summary))
            sys.stdout.close()
        except FileNotFoundError:
            os.makedirs(os.path.join(output, "parameters/"))
            sys.stdout = open(
                os.path.join(output, f"parameters/{cfg.model.name}.txt"), "w"
            )
            print(model.summarize(mode=trainer.weights_summary))
            sys.stdout.close()

        # GPU Memory logging (NOT YET IMPLETMENED)

    # train!
    trainer.fit(model, train_loader, val_loader)

    return model
