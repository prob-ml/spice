import pathlib

import numpy as np
import torch
import torch_geometric
from hydra.experimental import compose, initialize
from numpy import random as npr


class SimulatedData(torch_geometric.data.InMemoryDataset):
    def __init__(self, n_samples):
        super().__init__()

        data_list, self.data_dimension = simulate_data(n_samples)
        self.features = [0]
        self.responses = [1]
        self.data, self.slices = self.collate(data_list)


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
        expr = np.zeros((n_nodes, data_dimension))
        for i in range(expr.shape[0]):
            for j in range(expr.shape[1]):
                expr[i, j] = 2.2 * npr.randn() + i + j

        # random celltypes FIX THIS
        celltypes = npr.randint(0, 15, n_nodes)

        datalist.append(
            torch_geometric.data.Data(
                x=torch.tensor(expr.astype(np.float32)),
                edge_index=edges,
                pos=torch.tensor(pos.astype(np.float32)),
                y=torch.tensor(celltypes.astype(np.int32)),
            )
        )

    return datalist, data_dimension


def test_merfish_dataset():
    from spatial import merfish_dataset

    test_dir = pathlib.Path(__file__).parent.absolute()
    test_data_dir = test_dir.joinpath("data")
    # relative path needed to pass test on Github
    # maybe change pytest.ini so the relpath is shorter?
    mfd = merfish_dataset.MerfishDataset(
        test_data_dir, non_response_genes_file="../spatial/spatial/non_response.txt"
    )
    mfd.get(0)


# def simulate_data(n_samples):
#     datalist = []

#     # this problem should be pretty easy!
#     data_dimension = 2

#     # set random seed
#     npr.seed(0)
#     torch.manual_seed(0)

#     # generate random graphs
#     for _ in range(n_samples):
#         # random (but varying!) number of nodes
#         n_nodes = npr.randint(10, 20)

#         # fantasize some edges
#         edge_matrix = npr.rand(n_nodes, n_nodes) < 0.2
#         edge_matrix[np.r_[0:n_nodes], np.r_[0:n_nodes]] = False
#         edges = torch.tensor(np.array(np.where(edge_matrix)))

#         # 2d positions
#         pos = npr.randn(n_nodes, 2)

#         # random data
#         expr = np.zeros((n_nodes, data_dimension))
#         for i in range(expr.shape[0]):
#             for j in range(expr.shape[1]):
#                 expr[i, j] = 2.2 * npr.randn() + i + j

#         # random celltypes FIX THIS
#         celltypes = npr.randint(0, 15, n_nodes)

#         datalist.append(
#             torch_geometric.data.Data(
#                 x=torch.tensor(expr.astype(np.float32)),
#                 edge_index=edges,
#                 pos=torch.tensor(pos.astype(np.float32)),
#                 y=torch.tensor(celltypes.astype(np.int32)),
#             )
#         )

#     return datalist, data_dimension


def test_monetae2d():
    from spatial import predict, train

    ###################
    # make simulation
    data_dimension = SimulatedData(1).data_dimension
    train_simulated_data = SimulatedData(12)
    test_simulated_data = SimulatedData(2)

    assert data_dimension == 2

    ###################
    # fitmodel with updated hydra config

    overrides_train = {
        "gpus": 0,
        "model": "MonetAutoencoder2D",
        "model.name": "MonetAutoencoder2D",
        "model.label": "gitpush2",
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
    train_loss = trained_model.calc_loss(
        recon, train_simulated_data[0].x, cfg.model.kwargs.loss_type
    )

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
            predict.test(cfg, data=test_simulated_data)[0]
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
    import string
    import random

    ###################
    # make simulation
    data_dimension = SimulatedData(1).data_dimension
    train_simulated_data = SimulatedData(12)
    test_simulated_data = SimulatedData(2)

    ###################
    # fitmodel with updated hydra config

    overrides_train = {
        "gpus": 0,
        "model": "TrivialAutoencoder",
        "model.name": "TrivialAutoencoder",
        "model.label": "notebook",
        "model.kwargs.observables_dimension": data_dimension,
        "model.kwargs.hidden_dimensions": [100, 50, 25, 10],
        "model.kwargs.latent_dimension": 2,
        "training.n_epochs": 10,
        "training.logger_name": "".join(
            random.choices(string.ascii_uppercase + string.digits, k=12)
        ),
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
    train_loss = trained_model.calc_loss(
        recon, train_simulated_data[0].x, cfg.model.kwargs.loss_type
    )

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
            predict.test(cfg, data=test_simulated_data)[0]
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
    assert monet_train_loss < 0.98 * trivial_train_loss
    assert monet_test_loss < 0.98 * trivial_test_loss
