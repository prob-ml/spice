import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from spatial.models import base_networks

_models = [base_networks.DenseReluGMMConvNetwork]
models = {cls.__name__: cls for cls in _models}

# specify logger (taken from bliss)
def setup_logger(cfg):
    logger = False
    if cfg.training.trainer.logger:
        logger = TensorBoardLogger(
            save_dir=cfg.paths.root.output, name=cfg.trainer.logger.name
        )
    return logger


# set up model saving (taken from bliss)
def setup_checkpoint_callback(cfg, logger):
    checkpoint_callback = False
    output = cfg.paths.root.output
    if cfg.training.trainer.checkpoint_callback:
        checkpoint_dir = f"lightning_logs/version_{logger.version}/checkpoints"
        checkpoint_dir = output.joinpath(checkpoint_dir)
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
    model = models[cfg.model.name](cfg)

    # setup trainer
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, checkpoint_callback=checkpoint_callback))
    trainer = pl.Trainer(**trainer_dict)

    # train!
    trainer.fit(model)
