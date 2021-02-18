import types

import h5py
import numpy as np
import pandas as pd
import requests
import sklearn.neighbors
import torch
import torch_geometric


class MerfishDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, n_neighbors=3, train=True):
        super().__init__(root)

        data_list = self.construct_graphs(n_neighbors)

        with h5py.File(self.raw_dir + "/merfish.hdf5", "r") as h5f:
            self.gene_names = h5f["gene_names"][:].astype("U")[~self.bad_genes]

        # we use the first 150 slices for training
        if train:
            self.data, self.slices = self.collate(data_list[:150])
        else:
            self.data, self.slices = self.collate(data_list[150:])

    url = "https://datadryad.org/stash/downloads/file_stream/68364"

    behavior_types = [
        "Naive",
        "Parenting",
        "Virgin Parenting",
        "Aggression to pup",
        "Aggression to adult",
        "Mating",
    ]
    behavior_lookup = {x: i for (i, x) in enumerate(behavior_types)}

    bad_genes = np.zeros(161, dtype=np.bool)
    bad_genes[144] = True

    @property
    def raw_file_names(self):
        return ["merfish.csv", "merfish.hdf5"]

    def download(self):
        with open(self.raw_dir + "/merfish.csv", "wb") as csvf:
            csvf.write(requests.get(self.url).content)

        dataframe = pd.read_csv(self.raw_dir + "/merfish.csv")

        with h5py.File(self.raw_dir + "/merfish.hdf5", "w") as h5f:
            for colnm, dtype in zip(dataframe.keys()[:9], dataframe.dtypes[:9]):
                if dtype.kind == "O":
                    h5f.create_dataset(
                        colnm, data=np.require(dataframe[colnm], dtype="S36")
                    )
                else:
                    h5f.create_dataset(colnm, data=np.require(dataframe[colnm]))
            h5f.create_dataset(
                "expression",
                data=np.array(dataframe[dataframe.keys()[9:]]).astype(np.float16),
            )
            h5f.create_dataset(
                "gene_names", data=np.array(dataframe.keys()[9:], dtype="S80")
            )

    def construct_graph(self, data, anid, breg, n_neighbors):
        # get subset of cells in this slice
        good = (data.anids == anid) & (data.bregs == breg)

        # figure out neighborhood structure
        locations_for_this_slice = data.locations[good]
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="ball_tree"
        )
        nbrs.fit(locations_for_this_slice)
        _, neighbors = nbrs.kneighbors(locations_for_this_slice)
        edges = np.concatenate(
            [np.c_[neighbors[:, 0], neighbors[:, i + 1]] for i in range(n_neighbors)],
            axis=0,
        )
        edges = torch.tensor(edges, dtype=torch.long).T

        # remove gene 144.  which is bad.  for some reason.
        subexpression = data.expression[good]
        subexpression = subexpression[:, ~self.bad_genes]

        # get behavior ids
        behavior_ids = np.array([self.behavior_lookup[x] for x in data.behavior[good]])

        # make it into a torch geometric data object, add it to the list!
        return torch_geometric.data.Data(
            x=torch.tensor(subexpression.astype(np.float32)),
            edge_index=edges,
            pos=torch.tensor(locations_for_this_slice.astype(np.float32)),
            y=torch.tensor(behavior_ids),
        )

    def construct_graphs(self, n_neighbors):
        # load hdf5
        with h5py.File(self.raw_dir + "/merfish.hdf5", "r") as h5f:
            data = types.SimpleNamespace(
                anids=h5f["Animal_ID"][:],
                bregs=h5f["Bregma"][:],
                expression=h5f["expression"][:],
                locations=np.c_[h5f["Centroid_X"][:], h5f["Centroid_Y"][:]],
                behavior=h5f["Behavior"][:].astype("U"),
            )

        # get the (animal_id,bregma) pairs that define a unique slice
        unique_slices = np.unique(np.c_[data.anids, data.bregs], axis=0)

        # store all the slices in this list...
        data_list = []
        for anid, breg in unique_slices:
            data_list.append(self.construct_graph(data, anid, breg, n_neighbors))

        return data_list
