import os
import sys

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from spatial.merfish_dataset import MerfishDataset
from spatial.models import monet_ae

_models = [monet_ae.TrivialAutoencoder, monet_ae.MonetAutoencoder2D]
models = {cls.__name__: cls for cls in _models}

# specify logger (taken from bliss)
def setup_logger(cfg):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=cfg.paths.output, name=cfg.training.logger_name
        )
    return logger


# set up model saving (taken from bliss)
def setup_checkpoint_callback(cfg, logger):
    checkpoint_callback = False
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
        )

    return checkpoint_callback


def train(cfg: DictConfig):

    # setup logger
    logger = setup_logger(cfg)

    # setup checkpoints
    checkpoint_callback = setup_checkpoint_callback(cfg, logger)

    # specify model
    model = models[cfg.model.name](**cfg.model.kwargs)

    # setup training data
    data = MerfishDataset(cfg.paths.data, train=True)
    n_data = len(data)
    train_n = round(n_data * 11 / 12)
    train_data, val_data = random_split(data, [train_n, n_data - train_n])

    train_loader = DataLoader(train_data, batch_size=4, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=4, num_workers=2)
    print(len(train_loader))
    print(train_loader)

    # setup trainer
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(
        dict(
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=pl.callbacks.progress.ProgressBar().enable(),
        )
    )
    trainer = pl.Trainer(**trainer_dict)

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
