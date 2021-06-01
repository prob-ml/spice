import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric
from numpy import random as npr


def test_merfish_dataset():
    from spatial import merfish_dataset

    test_dir = pathlib.Path(__file__).parent.absolute()
    test_data_dir = test_dir.joinpath("data")
    mfd = merfish_dataset.MerfishDataset(test_data_dir)
    mfd.get(0)


def simulate_data():
    datalist = []

    # this problem should be pretty easy!
    data_dimension = 2

    # set random seed
    npr.seed(0)
    torch.manual_seed(0)

    # generate random graphs
    for _ in range(4):
        # random (but varying!) number of nodes
        n_nodes = npr.randint(10, 20)

        # fantasize some edges
        edge_matrix = npr.rand(n_nodes, n_nodes) < 0.2
        edge_matrix[np.r_[0:n_nodes], np.r_[0:n_nodes]] = False
        edges = torch.tensor(np.array(np.where(edge_matrix)))

        # 2d positions
        pos = npr.randn(n_nodes, 2)

        # random data
        expr = npr.randn(n_nodes, data_dimension)

        datalist.append(
            torch_geometric.data.Data(
                x=torch.tensor(expr.astype(np.float32)),
                edge_index=edges,
                pos=torch.tensor(pos.astype(np.float32)),
            )
        )

    return datalist, data_dimension


def test_monetae2d():
    from spatial.models import monet_ae

    ###################
    # make simulation
    simulated_data, data_dimension = simulate_data()

    ###################
    # fitmodel

    hidden_dimensions = [100, 50, 25, 10]
    latent_dimension = 2

    model = monet_ae.MonetAutoencoder2D(
        data_dimension,
        hidden_dimensions,
        latent_dimension,
        dim=2,
        kernel_size=25,
        loss_type="mse",
    )
    trainer = pl.Trainer(gpus=1, max_epochs=5)

    trainer.fit(
        model,
        torch_geometric.data.DataLoader(
            simulated_data[:2], batch_size=1, num_workers=2, pin_memory=True
        ),
        torch_geometric.data.DataLoader(
            simulated_data[2:], batch_size=1, shuffle=False, pin_memory=True
        ),
    )

    ###################
    # check our performance on a training example

    # get loss if our network outputs all zeros
    loss_if_we_are_dumb = np.sum(simulated_data[0].x.detach().cpu().numpy() ** 2)

    # get loss from trained network
    _, recon = model(simulated_data[0])
    loss = model.calc_loss(recon, simulated_data[0].x)

    # make sure we are doing better than the trivial answer
    assert loss.detach().cpu() < 0.3 * loss_if_we_are_dumb

    return loss


def test_trivial():
    from spatial.models import monet_ae

    ###################
    # make simulation
    simulated_data, data_dimension = simulate_data()

    ###################
    # fitmodel

    hidden_dimensions = [100, 50, 25, 10]
    latent_dimension = 2

    model = monet_ae.TrivialAutoencoder(
        data_dimension,
        hidden_dimensions,
        latent_dimension,
        loss_type="mse",
    )
    trainer = pl.Trainer(gpus=1, max_epochs=5)

    trainer.fit(
        model,
        torch_geometric.data.DataLoader(
            simulated_data[:2], batch_size=1, num_workers=2, pin_memory=True
        ),
        torch_geometric.data.DataLoader(
            simulated_data[2:], batch_size=1, shuffle=False, pin_memory=True
        ),
    )

    ###################
    # check our performance on a training example

    # get loss if our network outputs all zeros
    loss_if_we_are_dumb = np.sum(simulated_data[0].x.detach().cpu().numpy() ** 2)

    # get loss from trained network
    _, recon = model(simulated_data[0])
    loss = model.calc_loss(recon, simulated_data[0].x)

    # make sure we are doing better than the trivial answer
    assert loss.detach().cpu() < 0.4 * loss_if_we_are_dumb

    return loss


def test_accuracy():
    assert test_monetae2d() < test_trivial()
