import os
import numpy as np
import scipy.spatial
import pandas as pd
import itertools
import torch
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GMMConv
from torch_geometric.nn import avg_pool
from torch_geometric.nn import max_pool
from torch_geometric.nn import graclus
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.nn import max_pool_x
from torch_geometric.nn import global_mean_pool
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors


class Merfish(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, train=True):
        super(Merfish, self).__init__(root, transform, pre_transform)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['train.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def download(self):
        pass # for now
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.

        def cell_graph(merfish, animal, bregma):
            merfish = merfish[(merfish['Animal_ID'] == animal) & (merfish['Bregma'] == bregma)]
            if len(merfish.index) == 0:
                return "No Data"
            print("Animal: {0}, Bregma: {1}, Cells: {2}".format(animal, bregma, len(merfish.index)))
            pos = pd.DataFrame(merfish, columns=["Centroid_X", "Centroid_Y"]).to_numpy()
            x = torch.tensor(merfish.iloc[:, 11:].to_numpy())
            cell_id = pd.DataFrame(merfish, columns=["Cell_ID"]).to_numpy()
            celltype = pd.DataFrame(merfish, columns=["Cell_class"]).to_numpy()
            behavior = pd.DataFrame(merfish, columns=["Behavior"])
            behavior = behavior.iloc[:, 0].map({"Naive": 0, "Parenting": 1, "Virgin Parenting": 2,
                                                "Aggression to pup": 3, "Aggression to adult": 4,
                                                "Mating": 5})
            behavior = torch.tensor(behavior.to_numpy())
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(pos)
            distances, neighbors = nbrs.kneighbors(pos)
            pairwise_neighbors = list(map(lambda x: list(itertools.product([x[0]], x[1:])), neighbors))
            edge_index = torch.tensor(list(itertools.chain.from_iterable(pairwise_neighbors))).T
            return (x, cell_id, celltype, behavior, edge_index, pos)

        data_list = []
        merfish_df = pd.read_csv("../data/merfish.csv")
        for i in range(35):
            for j in (0.26, 0.21, 0.16, 0.11, 0.06, 0.01, -0.04, -0.09, -0.14, -0.19, -0.24, -0.29):
                try:
                    x, cell_id, celltype, behavior, edge_index, pos = cell_graph(merfish_df, i+1, j)
                    animal_data = Data(x=x, edge_index=edge_index,
                                       pos=torch.tensor(pos), y=behavior)
                    data_list.append(animal_data)
                except ValueError:
                    print("No data found for (Animal_ID: {0}, Bregma: {1})".format(i+1, j))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        n = len(data_list)
        train_n = round(n*6/7)
        print(train_n)
        train_list, test_list = random_split(data_list, [train_n, n - train_n])
        train_data, train_slices = self.collate(train_list)
        test_data, test_slices = self.collate(test_list)
        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((test_data, test_slices), self.processed_paths[1])

Merfish("../data")