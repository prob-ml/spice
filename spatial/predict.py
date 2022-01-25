from hydra.utils import instantiate

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch_geometric.data import DataLoader

from spatial.models.monet_ae import (
    MeanExpressionNN,
    MonetAutoencoder2D,
    TrivialAutoencoder,
)
from spatial.train import (
    setup_checkpoint_callback,
    setup_logger,
)


def test(cfg: DictConfig, data=None):

    # Set up testing data.
    if data is None:
        data = instantiate(cfg.datasets.dataset, train=False)

    # ensuring data dimension is correct
    if cfg.model.kwargs.observables_dimension != data[0].x.shape[1]:
        raise AssertionError("Data dimension not in line with observables dimension.")

    # get response indeces so they can be passed into the model
    if data.responses is not None:
        OmegaConf.update(cfg, "model.kwargs.responses", data.responses)

    # FOR NOW I NEED THIS TO KEEP TABS ON TESTING LOSS
    # setup logger
    logger = setup_logger(cfg)

    # setup checkpoints
    checkpoint_callback = setup_checkpoint_callback(cfg, logger)

    # get string of checkpoint path (FOR NEW RUNS)
    # pylint: disable=protected-access
    if cfg.datasets.dataset._target_.split(".")[-1] == "FilteredMerfishDataset":

        checkpoint_path = (
            f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.name}/{cfg.model.name}__"
            f"{cfg.model.kwargs.observables_dimension}"
            f"__{cfg.model.kwargs.hidden_dimensions}__"
            f"{cfg.model.kwargs.latent_dimension}__{cfg.n_neighbors}"
            f"__{cfg.datasets.dataset.sexes}__{cfg.datasets.dataset.behaviors}"
            f"__{cfg.optimizer.params.lr}__{cfg.training.logger_name}.ckpt"
        )

    else:

        checkpoint_path = (
            f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.name}/{cfg.model.name}__"
            f"{cfg.model.kwargs.observables_dimension}"
            f"__{cfg.model.kwargs.hidden_dimensions}__"
            f"{cfg.model.kwargs.latent_dimension}__{cfg.n_neighbors}"
            f"__{cfg.optimizer.params.lr}__{cfg.training.logger_name}.ckpt"
        )

    # get string of checkpoint path (FOR OLD RUNS)
    # checkpoint_path = (
    #     f"{cfg.paths.output}/lightning_logs/"
    #     f"checkpoints/{cfg.model.name}/{cfg.model.label}.ckpt"
    # )

    # Load the best model.
    if cfg.model.name == "TrivialAutoencoder":
        model = TrivialAutoencoder.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
        )
    if cfg.model.name == "MonetAutoencoder2D":
        model = MonetAutoencoder2D.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
        )
    if cfg.model.name == "MeanExpressionNN":
        model = MeanExpressionNN.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **cfg.model.kwargs,
        )
        for batch in data:
            batch.x = batch.x[:, model.features]

    test_loader = DataLoader(data, batch_size=cfg.predict.batch_size, num_workers=2)

    # Create trainer.
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, callbacks=checkpoint_callback))
    trainer = pl.Trainer(**trainer_dict)

    test_results = trainer.test(model, test_loader, verbose=cfg.predict.verbose)

    if cfg.model.name == "MeanExpressionNN":
        return trainer, model.inputs, model.gene_expressions, model.celltypes

    l1_losses = F.l1_loss(model.inputs, model.gene_expressions)

    # first is needed for testing, the rest is for jupyter notebook exploration fun!
    return (
        trainer,
        l1_losses,
        model.inputs,
        model.gene_expressions,
        model.celltypes,
        test_results,
    )
