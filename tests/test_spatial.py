import pathlib

import numpy as np
import torch
import torch_geometric
from hydra.experimental import compose, initialize
from numpy import random as npr


def test_merfish_dataset():
    from spatial import merfish_dataset

    test_dir = pathlib.Path(__file__).parent.absolute()
    test_data_dir = test_dir.joinpath("data")
    mfd = merfish_dataset.MerfishDataset(test_data_dir)
    mfd.get(0)


def simulate_data(n_samples):
    datalist = []

    # this problem should be pretty easy!
    data_dimension = 2

    # set random seed
    npr.seed(0)
    torch.manual_seed(0)

    # generate random graphs
    for _ in range(n_samples):
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
    from spatial import predict, train

    ###################
    # make simulation
    train_simulated_data, data_dimension = simulate_data(12)
    test_simulated_data, data_dimension = simulate_data(2)

    ###################
    # fitmodel with updated hydra config

    overrides_train = {
        "gpus": 0,
        "model": "MonetAutoencoder2D",
        "model.label": "test-MonetAutoencoder2D",
        "model.kwargs.observables_dimension": data_dimension,
        "model.kwargs.hidden_dimensions": [100, 50, 25, 10],
        "model.kwargs.latent_dimension": 2,
        "training.n_epochs": 10,
    }
    overrides_train_list = [f"{k}={v}" for k, v in overrides_train.items()]

    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides_train_list)
        trained_model = train.train(cfg, data=train_simulated_data)

    ###################
    # check our performance on a training example

    # get loss if our network outputs all zeros
    loss_if_we_are_dumb = np.sum(train_simulated_data[0].x.detach().cpu().numpy() ** 2)

    # get loss from trained network
    _, recon = trained_model(train_simulated_data[0])
    train_loss = trained_model.calc_loss(recon, train_simulated_data[0].x)

    # check if test loss is lower than dummy loss
    overrides_test = overrides_train.copy()
    test_addons = {
        "mode": "predict",
        "predict.verbose": False,
    }
    overrides_test.update(test_addons)

    overrides_test_list = [f"{k}={v}" for k, v in overrides_test.items()]

    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides_test_list)
        test_loss = (
            predict.test(cfg, data=test_simulated_data)
            .callback_metrics["test_loss"]
            .item()
        )

    # make sure we are doing better than the trivial answer
    # train case
    assert train_loss.detach().cpu() < 0.6 * loss_if_we_are_dumb
    # test case
    assert test_loss < 0.6 * loss_if_we_are_dumb

    return train_loss, test_loss


def test_trivial():
    from spatial import predict, train

    ###################
    # make simulation
    train_simulated_data, data_dimension = simulate_data(12)
    test_simulated_data, data_dimension = simulate_data(2)

    ###################
    # fitmodel with updated hydra config

    overrides_train = {
        "gpus": 0,
        "model": "TrivialAutoencoder",
        "model.label": "test-TrivialAutoencoder",
        "model.kwargs.observables_dimension": data_dimension,
        "model.kwargs.hidden_dimensions": [100, 50, 25, 10],
        "model.kwargs.latent_dimension": 2,
        "training.n_epochs": 10,
    }
    overrides_train_list = [f"{k}={v}" for k, v in overrides_train.items()]

    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides_train_list)
        trained_model = train.train(cfg, data=train_simulated_data)

    ###################
    # check our performance on a training example

    # get loss if our network outputs all zeros
    loss_if_we_are_dumb = np.sum(train_simulated_data[0].x.detach().cpu().numpy() ** 2)

    # get loss from trained network
    _, recon = trained_model(train_simulated_data[0])
    train_loss = trained_model.calc_loss(recon, train_simulated_data[0].x)

    # check if test loss is lower than dummy loss
    overrides_test = overrides_train.copy()
    test_addons = {
        "mode": "predict",
        "predict.verbose": False,
    }
    overrides_test.update(test_addons)

    overrides_test_list = [f"{k}={v}" for k, v in overrides_test.items()]

    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides_test_list)
        test_loss = (
            predict.test(cfg, data=test_simulated_data)
            .callback_metrics["test_loss"]
            .item()
        )

    # make sure we are doing better than the trivial answer
    # train case
    assert train_loss.detach().cpu() < 0.6 * loss_if_we_are_dumb
    # test case
    assert test_loss < 0.6 * loss_if_we_are_dumb

    return train_loss, test_loss


def test_accuracy():
    monet_train_loss, monet_test_loss = test_monetae2d()
    trivial_train_loss, trivial_test_loss = test_trivial()
    assert monet_train_loss < 0.85 * trivial_train_loss
    assert monet_test_loss < 0.85 * trivial_test_loss
