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
    def __init__(
        self,
        root,
        n_neighbors=3,
        train=True,
        log_transform=True,
        non_response_genes_file="/home/roko/spatial/spatial/non_response.txt",
    ):
        super().__init__(root)

        # non-response genes (columns) in MERFISH
        with open(non_response_genes_file, "r") as genes_file:
            self.features = [int(x) for x in genes_file.read().split(",")]
            genes_file.close()

        # response genes (columns in MERFISH)
        self.responses = list(set(range(160)) - set(self.features))

        data_list = self.construct_graphs(n_neighbors, train, log_transform)

        with h5py.File(self.merfish_hdf5, "r") as h5f:
            self.gene_names = h5f["gene_names"][:][~self.bad_genes].astype("U")

        self.data, self.slices = self.collate(data_list)

    # from https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248
    url = "https://datadryad.org/stash/downloads/file_stream/67671"

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

    @property
    def merfish_csv(self):
        return os.path.join(self.raw_dir, "merfish.csv")

    @property
    def merfish_hdf5(self):
        return os.path.join(self.raw_dir, "merfish.hdf5")

    def download(self):
        # download csv if necessary
        if not os.path.exists(self.merfish_csv):
            with open(self.merfish_csv, "wb") as csvf:
                csvf.write(requests.get(self.url).content)

        # process csv if necessary
        dataframe = pd.read_csv(self.merfish_csv)

        with h5py.File(self.merfish_hdf5, "w") as h5f:
            for colnm, dtype in zip(dataframe.keys()[:9], dataframe.dtypes[:9]):
                if dtype.kind == "O":
                    data = np.require(dataframe[colnm], dtype="S36")
                    h5f.create_dataset(colnm, data=data)
                else:
                    h5f.create_dataset(colnm, data=np.require(dataframe[colnm]))

            expression = np.array(dataframe[dataframe.keys()[9:]]).astype(np.float16)
            h5f.create_dataset("expression", data=expression)

            gene_names = np.array(dataframe.keys()[9:], dtype="S80")
            h5f.create_dataset("gene_names", data=gene_names)

    def construct_graph(self, data, anid, breg, n_neighbors, log_transform):
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

        # if we want to first log transform the data, we do it here
        # make this one return statement only changing x
        predictors_x = torch.tensor(subexpression.astype(np.float32))
        if log_transform:
            predictors_x = torch.log1p(predictors_x)

        return torch_geometric.data.Data(
            x=predictors_x,
            edge_index=edges,
            pos=torch.tensor(locations_for_this_slice.astype(np.float32)),
            y=torch.tensor(labelinfo),
        )

    def construct_graphs(self, n_neighbors, train, log_transform=True):
        # load hdf5
        with h5py.File(self.merfish_hdf5, "r") as h5f:
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
            data_list.append(
                self.construct_graph(data, anid, breg, n_neighbors, log_transform)
            )

        return data_list


class FilteredMerfishDataset(MerfishDataset):
    def __init__(
        self,
        root,
        n_neighbors=3,
        train=True,
        log_transform=True,
        non_response_genes_file="/home/roko/spatial/spatial/non_response.txt",
        sexes=None,
        behaviors=None,
    ):
        self.root = root
        self.sexes = sexes
        self.behaviors = behaviors
        original_csv_file = super().merfish_csv
        new_df = pd.read_csv(original_csv_file)
        print(f"Original Data {new_df.shape}")
        if self.sexes is not None:
            new_df = new_df[new_df["Animal_sex"].isin(self.sexes)]
        if self.behaviors is not None:
            new_df = new_df[new_df["Behavior"].isin(self.behaviors)]
        if new_df.shape[0] == 0:
            raise ValueError("Dataframe has no rows. Cannot build graph.")
        new_df.to_csv(self.root + "/raw/merfish_messi.csv", index=False)
        print(f"Filtered Data {new_df.shape}")
        # print("Filtered csv file created!")
        MerfishDataset.download(self)
        super().__init__(
            root,
            n_neighbors=n_neighbors,
            train=train,
            log_transform=log_transform,
            non_response_genes_file=non_response_genes_file,
        )
        # print("Filtered hdf5 file created!")

    #     @property
    #     def raw_file_names(self):
    #         return ["merfish_messi.csv", "merfish_messi.hdf5"]

    # THIS LINE WAS EDITED TO SHOW NEW FILE
    @property
    def merfish_csv(self):
        return os.path.join(self.raw_dir, "merfish_messi.csv")

    # THIS LINE WAS EDITED TO SHOW NEW FILE
    @property
    def merfish_hdf5(self):
        return os.path.join(self.raw_dir, "merfish_messi.hdf5")

    def construct_graphs(self, n_neighbors, train, log_transform=True):
        print(self.merfish_hdf5)
        # load hdf5
        with h5py.File(self.merfish_hdf5, "r") as h5f:
            # pylint: disable=no-member
            data = types.SimpleNamespace(
                anids=h5f["Animal_ID"][:],
                bregs=h5f["Bregma"][:],
                expression=h5f["expression"][:],
                locations=np.c_[h5f["Centroid_X"][:], h5f["Centroid_Y"][:]],
                behavior=h5f["Behavior"][:].astype("U"),
                celltypes=h5f["Cell_class"][:].astype("U"),
            )

        anid_to_bregma_count = {
            1: 12,
            2: 12,
            3: 6,
            4: 5,
            5: 6,
            6: 6,
            7: 12,
            8: 6,
            9: 6,
            10: 6,
            11: 6,
            12: 4,
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
            18: 4,
            19: 4,
            20: 4,
            21: 4,
            22: 4,
            23: 4,
            24: 4,
            25: 4,
            26: 4,
            27: 2,
            28: 4,
            29: 4,
            30: 4,
        }

        # get the (animal_id,bregma) pairs that define a unique slice
        unique_slices = np.unique(np.c_[data.anids, data.bregs], axis=0)

        # are we looking at train or test sets?
        min_animal = anid_to_bregma_count[np.min(data.anids)]
        unique_slices = (
            unique_slices[min_animal:] if train else unique_slices[:min_animal]
        )

        # store all the slices in this list...
        data_list = []
        for anid, breg in unique_slices:
            data_list.append(
                self.construct_graph(data, anid, breg, n_neighbors, log_transform)
            )

        return data_list
