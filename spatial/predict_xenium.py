# import datetime
import json

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate

# from torch.nn import functional as F
from torch_geometric.data import DataLoader

from spatial.models.monet_ae import (
    MonetAutoencoder2D,
    TrivialAutoencoder,
    MonetDense,
    TrivialDense,
    # MonetVAE,
    # GraphUNetDense,
)
from spatial.train import (
    setup_checkpoint_callback,
    setup_logger,
    setup_optimizer,
)

# pylint: disable=too-many-branches
def test(cfg: DictConfig, data=None):

    # Set up testing data.
    if data is None:
        if isinstance(cfg.datasets.dataset, list):
            data = instantiate(cfg.datasets.dataset[0], train=False)
        else:
            data = instantiate(cfg.datasets.dataset, train=False)
    # ensuring data dimension is correct
    # FIX THIS TO BE IN LINE WITH THE MODEL AT HAND
    # if cfg.model.kwargs.observables_dimension != data[0].x.shape[1]:
    #     raise AssertionError("Data dimension not in line with observables dimension.")

    # get response indeces so they can be passed into the model
    if cfg.model.kwargs.response_genes is None:
        OmegaConf.update(cfg, "model.kwargs.response_genes", data.response_genes)

    # FOR NOW I NEED THIS TO KEEP TABS ON TESTING LOSS
    # setup logger
    logger = setup_logger(cfg, filepath=cfg.training.filepath)

    # setup checkpoints
    checkpoint_callback = setup_checkpoint_callback(
        cfg, logger, filepath=cfg.training.filepath
    )

    # setup optimizer
    optimizer = setup_optimizer(cfg)

    checkpoint_path = (
        f"{cfg.paths.output}/lightning_logs/checkpoints/{cfg.model.name}Xenium/"
        + cfg.predict.filepath
        + ".ckpt"
    )

    # Load the best model.
    if cfg.model.name == "MonetDense":

        model = MonetDense.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
            optimizer=optimizer,
        )
    if cfg.model.name == "TrivialDense":
        model = TrivialDense.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
            optimizer=optimizer,
        )
    if cfg.model.name == "TrivialAutoencoder":
        model = TrivialAutoencoder.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
            optimizer=optimizer,
        )
    if cfg.model.name == "MonetAutoencoder2D":
        model = MonetAutoencoder2D.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
            optimizer=optimizer,
        )

    # Number of Parameters
    print(
        f"The number of parameters for radius {cfg.radius}"
        f" is {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    test_loader = DataLoader(data, batch_size=cfg.predict.batch_size, num_workers=8)

    # Create trainer.
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update({"logger": logger, "callbacks": checkpoint_callback})
    trainer = pl.Trainer(**trainer_dict)

    test_results = trainer.test(model, test_loader, verbose=cfg.predict.verbose)

    l1_losses = "currently unneeded"  # F.l1_loss(model.inputs, model.gene_expressions)

    with open(
        "/home/roko/spatial/scratch/recent_result.json", "w+", encoding="utf-8"
    ) as deepst_result:
        json.dump({"result": test_results[0]["test_loss"]}, deepst_result)

    # first is needed for testing, the rest is for jupyter notebook exploration fun!
    return (
        trainer,
        l1_losses,
        model.inputs,
        model.gene_expressions,
        model.celltypes,
        test_results,
    )
