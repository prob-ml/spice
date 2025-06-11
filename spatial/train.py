# pylint: disable=too-many-statements,too-many-branches

import os
import torch

from hydra.utils import instantiate

# import torch.profiler as profiler
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data.lightning import LightningDataset

# k fold
from sklearn.model_selection import StratifiedKFold

from spatial.merfish_dataset import FilteredMerfishDataset, MerfishDataset, DemoDataset
from spatial.models import monet_ae


_datasets = [FilteredMerfishDataset, MerfishDataset, DemoDataset]
datasets = {cls.__name__: cls for cls in _datasets}
_models = [
    monet_ae.TrivialAutoencoder,
    monet_ae.MonetAutoencoder2D,
    monet_ae.MonetDense,
    monet_ae.TrivialDense,
    monet_ae.MonetVAE,
    monet_ae.GraphUNetDense,
]
models = {cls.__name__: cls for cls in _models}

# specify logger (taken from bliss)
def setup_logger(cfg, filepath):
    logger = False
    if cfg.training.trainer.logger:

        logger = TensorBoardLogger(
            save_dir=cfg.paths.output,
            name=cfg.training.logger_name,
            version=(filepath),
        )

    return logger


# set up model saving (taken from bliss)
def setup_checkpoint_callback(cfg, logger, filepath):
    callbacks = []
    if cfg.training.trainer.enable_checkpointing:
        checkpoint_dir = f"lightning_logs/checkpoints/{cfg.model.name}"
        checkpoint_dir = os.path.join(cfg.paths.output, checkpoint_dir)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            verbose=True,
            monitor="val_loss",
            mode="min",
            filename=filepath,
        )
        callbacks.append(checkpoint_callback)

    return callbacks


def setup_early_stopping(cfg, callbacks):
    if cfg.training.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            min_delta=cfg.training.early_stopping.min_delta,
            patience=cfg.training.early_stopping.patience,
            verbose=cfg.training.early_stopping.verbose,
            mode=cfg.training.early_stopping.mode,
        )
        callbacks.append(early_stop_callback)
    return callbacks


def setup_optimizer(cfg):
    return {"name": cfg.optimizer.name, "params": cfg.optimizer.params}


def k_fold_over_animals(animal_list, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    val_indices, train_indices = [], []
    for _, idx in skf.split(
        torch.zeros(len(animal_list)), torch.zeros(len(animal_list))
    ):
        val_indices.append(torch.from_numpy(idx).to(torch.long))

    for i in range(folds):
        train_mask = torch.ones(len(animal_list), dtype=torch.bool)
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, val_indices


def train(cfg: DictConfig, data=None, validate_only=False, lightning_integration=True):

    # if this is a non-zero int, the run will have a seed
    if cfg.training.seed:
        pl.seed_everything(cfg.training.seed)

    # setup training data
    if data is None:
        data = instantiate(cfg.datasets.dataset)

    # ensuring data dimension is correct
    # THE BELOW SHOULD BE REWRITTEN BASED ON AN UPDATED CRITERIA
    # if cfg.model.kwargs.observables_dimension != data[0].x.shape[1]:
    #     raise AssertionError("Data dimension not in line with observables dimension.")

    if hasattr(cfg.model, "attach_mask") and cfg.model.kwargs.attach_mask:
        OmegaConf.update(
            cfg,
            "model.kwargs.observables_dimension",
            cfg.model.kwargs.observables_dimension * 2,
        )

    # get response indeces so they can be passed into the model
    if cfg.model.kwargs.response_genes is None:
        OmegaConf.update(cfg, "model.kwargs.response_genes", data.response_genes)

    # setup optimizer
    optimizer = setup_optimizer(cfg)

    # setup logger
    logger = setup_logger(cfg, filepath=cfg.training.filepath)

    # setup checkpoints
    callbacks = setup_checkpoint_callback(cfg, logger, filepath=cfg.training.filepath)

    callbacks = setup_early_stopping(cfg, callbacks)

    # setup trainer
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(
        {
            "logger": logger,
            "callbacks": callbacks,
        }
    )
    trainer = pl.Trainer(**trainer_dict)

    # specify model
    model = models[cfg.model.name](**cfg.model.kwargs, optimizer=optimizer)

    if cfg.training.folds > 1:

        animal_ids = data.anid.unique()

        train_indices, val_indices = k_fold_over_animals(
            animal_ids, folds=cfg.training.folds
        )

        original_filepath = cfg.training.filepath

        for fold in range(cfg.training.folds):

            OmegaConf.update(
                cfg,
                "training.filepath",
                f"{original_filepath}__FOLD={fold}",
            )

            # setup logger
            logger = setup_logger(cfg, filepath=cfg.training.filepath)

            # setup checkpoints
            callbacks = setup_checkpoint_callback(
                cfg, logger, filepath=cfg.training.filepath
            )

            callbacks = setup_early_stopping(cfg, callbacks)

            # setup trainer
            trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
            trainer_dict.update(
                {
                    "logger": logger,
                    "callbacks": callbacks,
                }
            )
            trainer = pl.Trainer(**trainer_dict)

            # specify model
            model = models[cfg.model.name](**cfg.model.kwargs, optimizer=optimizer)

            train_animals = animal_ids[train_indices[fold]]
            val_animals = animal_ids[val_indices[fold]]

            train_data = [sample for sample in data if sample.anid in train_animals]
            val_data = [sample for sample in data if sample.anid in val_animals]

            train_loader = DataLoader(
                train_data, batch_size=cfg.training.batch_size, num_workers=2
            )
            val_loader = DataLoader(
                val_data, batch_size=cfg.training.batch_size, num_workers=2
            )

            # train or validate
            if lightning_integration:
                datamodule = LightningDataset(
                    train_dataset=train_data,
                    val_dataset=val_data,
                    batch_size=1,
                    num_workers=2,
                )
                if validate_only:
                    checkpoint_dir = f"lightning_logs/checkpoints/{cfg.model.name}"
                    checkpoint_dir = os.path.join(cfg.paths.output, checkpoint_dir)
                    ckpt_path_for_validation = os.path.join(
                        checkpoint_dir, cfg.training.filepath + ".ckpt"
                    )
                    trainer.validate(
                        model, val_loader, ckpt_path=ckpt_path_for_validation
                    )
                else:
                    trainer.fit(model, datamodule)

            else:
                if validate_only:
                    checkpoint_dir = f"lightning_logs/checkpoints/{cfg.model.name}"
                    checkpoint_dir = os.path.join(cfg.paths.output, checkpoint_dir)
                    ckpt_path_for_validation = os.path.join(
                        checkpoint_dir, cfg.training.filepath + ".ckpt"
                    )
                    trainer.validate(
                        model, val_loader, ckpt_path=ckpt_path_for_validation
                    )
                else:
                    trainer.fit(model, train_loader, val_loader)

            torch.cuda.empty_cache()

    else:

        train_data = [sample for sample in data if sample.anid != 30]
        val_data = [sample for sample in data if sample.anid == 30]

        train_loader = DataLoader(
            train_data, batch_size=cfg.training.batch_size, num_workers=2
        )
        val_loader = DataLoader(
            val_data, batch_size=cfg.training.batch_size, num_workers=2
        )

        # train or validate
        if lightning_integration:
            datamodule = LightningDataset(
                train_dataset=train_data,
                val_dataset=val_data,
                batch_size=1,
                num_workers=2,
            )
            if validate_only:
                checkpoint_dir = f"lightning_logs/checkpoints/{cfg.model.name}"
                checkpoint_dir = os.path.join(cfg.paths.output, checkpoint_dir)
                ckpt_path_for_validation = os.path.join(
                    checkpoint_dir, cfg.training.filepath + ".ckpt"
                )
                trainer.validate(model, val_loader, ckpt_path=ckpt_path_for_validation)
            else:
                trainer.fit(model, datamodule)

        else:
            if validate_only:
                checkpoint_dir = f"lightning_logs/checkpoints/{cfg.model.name}"
                checkpoint_dir = os.path.join(cfg.paths.output, checkpoint_dir)
                ckpt_path_for_validation = os.path.join(
                    checkpoint_dir, cfg.training.filepath + ".ckpt"
                )
                trainer.validate(model, val_loader, ckpt_path=ckpt_path_for_validation)
            else:
                trainer.fit(model, train_loader, val_loader)

    return model, trainer
