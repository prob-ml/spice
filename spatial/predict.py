import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import DataLoader

from spatial.merfish_dataset import MerfishDataset
from spatial.models.monet_ae import MonetAutoencoder2D, TrivialAutoencoder

from spatial.train import setup_checkpoint_callback, setup_logger


def test(cfg: DictConfig):

    # FOR NOW I NEED THIS TO KEEP TABS ON TESTING LOSS
    # setup logger
    logger = setup_logger(cfg)

    # setup checkpoints
    checkpoint_callback = setup_checkpoint_callback(cfg, logger)

    # Load the best model.
    if cfg.model.name == "TrivialAutoencoder":
        model = TrivialAutoencoder.load_from_checkpoint(
            checkpoint_path=f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.label}.ckpt",
            **cfg.model.kwargs,
        )
    if cfg.model.name == "MonetAutoencoder2D":
        model = MonetAutoencoder2D.load_from_checkpoint(
            checkpoint_path=f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.label}.ckpt",
            **cfg.model.kwargs,
        )

    # Set up testing data.

    test_loader = DataLoader(
        MerfishDataset(cfg.paths.data, train=False), batch_size=1, num_workers=2
    )

    # Create trainer.
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, checkpoint_callback=checkpoint_callback))
    trainer = pl.Trainer(**trainer_dict)

    # Return the testing loss and accuracy.
    trainer.test(model, test_loader)
