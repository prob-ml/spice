import os
import types

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric
from sklearn import neighbors


class MerfishDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, n_neighbors=3, train=True):
        super().__init__(root)

        data_list = self.construct_graphs(n_neighbors, train)

        with h5py.File(self.raw_dir + "/merfish.hdf5", "r") as h5f:
            # pylint: disable=no-member
            self.gene_names = h5f["gene_names"][:][~self.bad_genes].astype("U")

        self.data, self.slices = self.collate(data_list)

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
    cell_types = [
        "Ambiguous",
        "Astrocyte",
        "Endothelial 1",
        "Endothelial 2",
        "Endothelial 3",
        "Ependymal",
        "Excitatory",
        "Inhibitory",
        "Microglia",
        "OD Immature 1",
        "OD Immature 2",
        "OD Mature 1",
        "OD Mature 2",
        "OD Mature 3",
        "OD Mature 4",
        "Pericytes",
    ]
    celltype_lookup = {x: i for (i, x) in enumerate(cell_types)}

    bad_genes = np.zeros(161, dtype=bool)
    bad_genes[144] = True

    @property
    def raw_file_names(self):
        return ["merfish.csv", "merfish.hdf5"]

    def download(self):
        # download csv if necessary
        csvfn = self.raw_dir + "/merfish.csv"
        if not os.path.exists(csvfn):
            with open(csvfn, "wb") as csvf:
                csvf.write(requests.get(self.url).content)

        # process csv if necessary
        dataframe = pd.read_csv(csvfn)

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
        nbrs = neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, algorithm="ball_tree"
        )
        nbrs.fit(locations_for_this_slice)
        _, kneighbors = nbrs.kneighbors(locations_for_this_slice)
        edges = np.concatenate(
            [np.c_[kneighbors[:, 0], kneighbors[:, i + 1]] for i in range(n_neighbors)],
            axis=0,
        )
        edges = torch.tensor(edges, dtype=torch.long).T

        # remove gene 144.  which is bad.  for some reason.
        subexpression = data.expression[good]
        subexpression = subexpression[:, ~self.bad_genes]

        # get behavior ids
        behavior_ids = np.array([self.behavior_lookup[x] for x in data.behavior[good]])
        celltype_ids = np.array([self.celltype_lookup[x] for x in data.celltypes[good]])
        labelinfo = np.c_[behavior_ids, celltype_ids]

        # make it into a torch geometric data object, add it to the list!
        return torch_geometric.data.Data(
            x=torch.tensor(subexpression.astype(np.float32)),
            edge_index=edges,
            pos=torch.tensor(locations_for_this_slice.astype(np.float32)),
            y=torch.tensor(labelinfo),
        )

    def construct_graphs(self, n_neighbors, train):
        # load hdf5
        with h5py.File(self.raw_dir + "/merfish.hdf5", "r") as h5f:
            # pylint: disable=no-member
            data = types.SimpleNamespace(
                anids=h5f["Animal_ID"][:],
                bregs=h5f["Bregma"][:],
                expression=h5f["expression"][:],
                locations=np.c_[h5f["Centroid_X"][:], h5f["Centroid_Y"][:]],
                behavior=h5f["Behavior"][:].astype("U"),
                celltypes=h5f["Cell_class"][:].astype("U"),
            )

        # get the (animal_id,bregma) pairs that define a unique slice
        unique_slices = np.unique(np.c_[data.anids, data.bregs], axis=0)

        # are we looking at train or test sets?
        unique_slices = unique_slices[:150] if train else unique_slices[150:]

        # store all the slices in this list...
        data_list = []
        for anid, breg in unique_slices:
            data_list.append(self.construct_graph(data, anid, breg, n_neighbors))

        return data_list
