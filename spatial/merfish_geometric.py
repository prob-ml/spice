import itertools

import pandas as pd
import torch
import sklearn.neighbors
import torch_geometric
import h5py
import numpy as np

class Merfish(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, n_neighbors=3,train=True):
        super().__init__(root)

        data_list=self.construct_graphs(n_neighbors)

        # we use the first 150 slices for training
        if train:
            self.data, self.slices = self.collate(data_list[:150])
        else:
            self.data, self.slices = self.collate(data_list[150:])

    url='https://datadryad.org/stash/downloads/file_stream/68364'

    behavior_types = [
        "Naive",
        "Parenting",
        "Virgin Parenting",
        "Aggression to pup",
        "Aggression to adult",
        "Mating",
    ]

    @property
    def raw_file_names(self):
        return ["merfish.csv","merfish.hdf5"]


    def download(self):
        with open(self.raw_dir+'/merfish.csv','wb') as f:
            f.write(requests.get(url).content)

        df=pd.read_csv(self.raw_dir+"/merfish.csv")

        with h5py.File(self.raw_dir+'/merfish.hdf5','w') as f:
            for nm,dtype in zip(df.keys()[:9],df.dtypes[:9]):
                if dtype.kind=='O':
                    f.create_dataset(nm,data=np.require(df[nm],dtype='S36'))
                else:
                    f.create_dataset(nm,data=np.require(df[nm]))
            f.create_dataset('expression',data=np.array(df[df.keys()[9:]]).astype(np.float16))
            f.create_dataset('gene_names',data=np.array(df.keys()[9:],dtype='S80'))

    def construct_graphs(self,n_neighbors):
        # load hdf5
        with h5py.File(self.raw_dir+'/merfish.hdf5','r') as f:
            anids=f['Animal_ID'][:]
            bregs=f['Bregma'][:]
            expression=f['expression'][:]
            locations=np.c_[f['Centroid_X'][:],f['Centroid_Y'][:]]
            behavior=f['Behavior'][:].astype("U")

        behavior_lookup={x:i for (i,x) in enumerate(self.behavior_types)}
        behavior_ids=np.array([behavior_lookup[x] for x in behavior])

        # get the (animal_id,bregma) pairs that define a unique slice
        unique_slices=np.unique(np.c_[anids,bregs],axis=0)

        # store all the slices in this list...
        data_list=[]
        for anid,breg in unique_slices:
            # get subset of cells in this slice
            good=(anids==anid)&(bregs==breg)

            # figure out neighborhood structure
            locations_for_this_slice=locations[good]
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree')
            nbrs.fit(locations_for_this_slice)
            distances, neighbors = nbrs.kneighbors(locations_for_this_slice)
            edges=np.concatenate([np.c_[neighbors[:,0],neighbors[:,i+1]] for i in range(n_neighbors)],axis=0)
            edges=torch.tensor(edges,dtype=torch.long).T

            # make it into a torch geometric data object, add it to the list!
            data_list.append(torch_geometric.data.Data(
                x=expression[good],
                edge_index=edges,
                pos=locations_for_this_slice,
                y=behavior_ids[good]
            ))

        return data_list
