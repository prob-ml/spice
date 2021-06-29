import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import DataLoader

from spatial.merfish_dataset import MerfishDataset
from spatial.models.monet_ae import MonetAutoencoder2D, TrivialAutoencoder
from spatial.train import setup_checkpoint_callback, setup_logger


def test(cfg: DictConfig, data=None):

    # FOR NOW I NEED THIS TO KEEP TABS ON TESTING LOSS
    # setup logger
    logger = setup_logger(cfg)

    # setup checkpoints
    checkpoint_callback = setup_checkpoint_callback(cfg, logger)

    # Load the best model.
    if cfg.model.name == "TrivialAutoencoder":
        model = TrivialAutoencoder.load_from_checkpoint(
            checkpoint_path=f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.name}/{cfg.model.label}.ckpt",
            **cfg.model.kwargs,
        )
    if cfg.model.name == "MonetAutoencoder2D":
        model = MonetAutoencoder2D.load_from_checkpoint(
            checkpoint_path=f"{cfg.paths.output}/lightning_logs/"
            f"checkpoints/{cfg.model.name}/{cfg.model.label}.ckpt",
            **cfg.model.kwargs,
        )

    # Set up testing data.
    if data is None:
        data = MerfishDataset(cfg.paths.data, train=False)

    test_loader = DataLoader(data, batch_size=1, num_workers=2)

    # Create trainer.
    trainer_dict = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    trainer_dict.update(dict(logger=logger, callbacks=[checkpoint_callback]))
    trainer = pl.Trainer(**trainer_dict)

    # # Return the testing loss and accuracy.
    # try:
    #     trainer.test(model, test_loader)
    # except ValueError:
    trainer.test(model, test_loader, verbose=cfg.predict.verbose)

    l1_losses = abs(model.inputs - model.gene_expressions)

    import pandas as pd

    # maxes = {}
    # for i in range(160):
    #     stuff = pd.Series(L1_losses[:,i])
    #     print(round(stuff.describe()[-5:], 3))
    #     stuff = stuff.drop(stuff.idxmax())
    #     try:
    #         maxes[stuff.idxmax()] += 1
    #     except:
    #         maxes[stuff.idxmax()] = 1
    # print(maxes)
    # print("MESSI: " + str([0.37, 0.38, 0.387, 0.389, 0.392]))
    # return trainer

    non_response_genes = []
    all_pairs_columns = [
        "Ligand.ApprovedSymbol",
        "Receptor.ApprovedSymbol",
    ]
    df_file = pd.ExcelFile("~/spatial-main/data/messi.xlsx")
    messi_df = pd.read_excel(df_file, "All.Pairs")
    merfish_df = pd.read_csv("~/spatial-main/data/merfish.csv")
    print(messi_df["Ligand.ApprovedSymbol"])
    for column in all_pairs_columns:
        for gene in merfish_df.columns:
            if (
                gene.upper() in list(messi_df[column])
                and gene.upper() not in non_response_genes
            ):
                non_response_genes.append(gene)
    print(non_response_genes)
    print(
        "There are "
        + str(len(non_response_genes))
        + " genes recognized as either ligands or receptors."
    )

    return l1_losses
