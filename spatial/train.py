import os

from hydra.utils import instantiate

# import torch.profiler as profiler
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from spatial.merfish_dataset import FilteredMerfishDataset, MerfishDataset
from spatial.models import monet_ae

_datasets = [FilteredMerfishDataset, MerfishDataset]
datasets = {cls.__name__: cls for cls in _datasets}
_models = [
    monet_ae.TrivialAutoencoder,
    monet_ae.MonetAutoencoder2D,
    monet_ae.MonetDense,
    monet_ae.TrivialDense,
]
models = {cls.__name__: cls for cls in _models}

# specify logger (taken from bliss)
def setup_logger(cfg, filepath):
    logger = False
    if cfg.training.trainer.logger:

        if cfg.model.name == "MonetAutoencoder2D":
            logger = TensorBoardLogger(
                save_dir=cfg.paths.output,
                name=cfg.training.logger_name,
                version=(filepath),
            )

        else:
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


def train(cfg: DictConfig, data=None, validate_only=False):

    # if this is a non-zero int, the run will have a seed
    if cfg.training.seed:
        pl.seed_everything(cfg.training.seed)

    # setup training data
    if data is None:
        data = instantiate(cfg.datasets.dataset)
    # n_data = len(data)
    # train_n = n_data - 1 if cfg.training.complete_data else round(n_data * 11 / 12)
    # train_data, val_data = random_split(
    #     data, [train_n, n_data - train_n], torch.Generator().manual_seed(42)
    # )
    train_data = [sample for sample in data if sample.anid != 30]
    val_data = [sample for sample in data if sample.anid == 30]

    train_loader = DataLoader(
        train_data, batch_size=cfg.training.batch_size, num_workers=2
    )
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size, num_workers=2)

    # ensuring data dimension is correct
    # THE BELOW SHOULD BE REWRITTEN BASED ON AN UPDATED CRITERIA
    # if cfg.model.kwargs.observables_dimension != data[0].x.shape[1]:
    #     raise AssertionError("Data dimension not in line with observables dimension.")

    if cfg.model.kwargs.attach_mask:
        OmegaConf.update(
            cfg,
            "model.kwargs.observables_dimension",
            cfg.model.kwargs.observables_dimension * 2,
        )

    # get response indeces so they can be passed into the model
    if cfg.model.kwargs.response_genes is None:
        OmegaConf.update(cfg, "model.kwargs.response_genes", data.response_genes)

    # get celltype lookup so we can filter by celltype in loss function
    # if data.celltype_lookup is not None:
    #     OmegaConf.update(cfg, "model.kwargs.celltype_lookup", data.celltype_lookup)

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

    # setup optimizer
    optimizer = setup_optimizer(cfg)

    # specify model
    model = models[cfg.model.name](**cfg.model.kwargs, optimizer=optimizer)

    # save model info
    # if cfg.training.save_model_summary:

    #     output = cfg.paths.output

    #     # architecture
    #     try:
    #         sys.stdout = open(
    #             os.path.join(output, f"architecture/{cfg.model.name}.txt"), "w"
    #         )
    #         print(model)
    #         sys.stdout.close()
    #     except FileNotFoundError:
    #         os.makedirs(os.path.join(output, "architecture/"))
    #         sys.stdout = open(
    #             os.path.join(output, f"architecture/{cfg.model.name}.txt"), "w"
    #         )
    #         print(model)
    #         sys.stdout.close()

    #     # parameters (and model memory size)
    #     try:
    #         sys.stdout = open(
    #             os.path.join(output, f"parameters/{cfg.model.name}.txt"), "w"
    #         )
    #         print(model.summarize(mode=trainer.weights_summary))
    #         sys.stdout.close()
    #     except FileNotFoundError:
    #         os.makedirs(os.path.join(output, "parameters/"))
    #         sys.stdout = open(
    #             os.path.join(output, f"parameters/{cfg.model.name}.txt"), "w"
    #         )
    #         print(model.summarize(mode=trainer.weights_summary))
    #         sys.stdout.close()

    # GPU Memory logging (NOT YET IMPLETMENED)

    # train or validate
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
