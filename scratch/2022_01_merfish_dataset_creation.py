# pylint: disable-all
import h5py
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg

rng = np.random.default_rng()
import tqdm.notebook
import pickle
import sys
import ipywidgets
import sklearn.neighbors
import json
import requests
from scipy.sparse import csr_matrix
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


original_url = "https://datadryad.org/stash/downloads/file_stream/67671"
csv_location = "data/spatial/moffit_merfish/original_file.csv"
h5ad_location = "data/spatial/moffit_merfish/original_file.h5ad"
connectivity_matrix_template = "data/spatial/moffit_merfish/connectivity_%d%s.h5ad"
genetypes_location = "data/spatial/moffit_merfish/genetypes.pkl"


# # download csv

# In[ ]:

with open(csv_location, "wb") as csvf:
    csvf.write(requests.get(original_url).content)


# # munge into hdf5 file

# In[ ]:


dataframe = pd.read_csv(csv_location)

dct = {}
for colnm, dtype in zip(dataframe.keys()[:9], dataframe.dtypes[:9]):
    if dtype.kind == "O":
        dct[colnm] = np.require(dataframe[colnm], dtype="U36")
    else:
        dct[colnm] = np.require(dataframe[colnm])
expression = np.array(dataframe[dataframe.keys()[9:]]).astype(np.float16)
gene_names = np.array(dataframe.keys()[9:], dtype="U80")
cellid = dct.pop("Cell_ID")

ad = anndata.AnnData(
    X=expression,
    var=pd.DataFrame(index=gene_names),
    obs=pd.DataFrame(dct, index=cellid),
)

ad.write_h5ad(h5ad_location)


# # supplement hdf5 file with a column indicating "tissue id" for each cell

# In[14]:


ad = anndata.read_h5ad(h5ad_location)
animal_ids = np.unique(ad.obs["Animal_ID"])
bregmas = np.unique(ad.obs["Bregma"])
tissue_id = np.zeros(len(ad), dtype=int)
n_tissues = 0

for aid in animal_ids:
    for bregma in bregmas:
        good = (ad.obs["Animal_ID"] == aid) & (ad.obs["Bregma"] == bregma)
        if np.sum(good) > 0:
            tissue_id[good] = n_tissues
            n_tissues += 1
ad.obs["Tissue_ID"] = tissue_id
ad.write_h5ad(h5ad_location)


# # write down ligand/receptor sets

# In[ ]:


ligands = np.array(
    [
        "Cbln1",
        "Cxcl14",
        "Cbln2",
        "Vgf",
        "Scg2",
        "Cartpt",
        "Tac2",
        "Bdnf",
        "Bmp7",
        "Cyr61",
        "Fn1",
        "Fst",
        "Gad1",
        "Ntng1",
        "Pnoc",
        "Selplg",
        "Sema3c",
        "Sema4d",
        "Serpine1",
        "Adcyap1",
        "Cck",
        "Crh",
        "Gal",
        "Gnrh1",
        "Nts",
        "Oxt",
        "Penk",
        "Sst",
        "Tac1",
        "Trh",
        "Ucn3",
    ]
)

receptors = np.array(
    [
        "Crhbp",
        "Gabra1",
        "Gpr165",
        "Glra3",
        "Gabrg1",
        "Adora2a",
        "Avpr1a",
        "Avpr2",
        "Brs3",
        "Calcr",
        "Cckar",
        "Cckbr",
        "Crhr1",
        "Crhr2",
        "Galr1",
        "Galr2",
        "Grpr",
        "Htr2c",
        "Igf1r",
        "Igf2r",
        "Kiss1r",
        "Lepr",
        "Lpar1",
        "Mc4r",
        "Npy1r",
        "Npy2r",
        "Ntsr1",
        "Oprd1",
        "Oprk1",
        "Oprl1",
        "Oxtr",
        "Pdgfra",
        "Prlr",
        "Ramp3",
        "Rxfp1",
        "Slc17a7",
        "Slc18a2",
        "Tacr1",
        "Tacr3",
        "Trhr",
    ]
)

response_genes = np.array(
    [
        "Ace2",
        "Aldh1l1",
        "Amigo2",
        "Ano3",
        "Aqp4",
        "Ar",
        "Arhgap36",
        "Baiap2",
        "Ccnd2",
        "Cd24a",
        "Cdkn1a",
        "Cenpe",
        "Chat",
        "Coch",
        "Col25a1",
        "Cplx3",
        "Cpne5",
        "Creb3l1",
        "Cspg5",
        "Cyp19a1",
        "Cyp26a1",
        "Dgkk",
        "Ebf3",
        "Egr2",
        "Ermn",
        "Esr1",
        "Etv1",
        "Fbxw13",
        "Fezf1",
        "Gbx2",
        "Gda",
        "Gem",
        "Gjc3",
        "Greb1",
        "Irs4",
        "Isl1",
        "Klf4",
        "Krt90",
        "Lmod1",
        "Man1a",
        "Mbp",
        "Mki67",
        "Mlc1",
        "Myh11",
        "Ndnf",
        "Ndrg1",
        "Necab1",
        "Nnat",
        "Nos1",
        "Npas1",
        "Nup62cl",
        "Omp",
        "Onecut2",
        "Opalin",
        "Pak3",
        "Pcdh11x",
        "Pgr",
        "Plin3",
        "Pou3f2",
        "Rgs2",
        "Rgs5",
        "Rnd3",
        "Scgn",
        "Serpinb1b",
        "Sgk1",
        "Slc15a3",
        "Slc17a6",
        "Slc17a8",
        "Slco1a4",
        "Sln",
        "Sox4",
        "Sox6",
        "Sox8",
        "Sp9",
        "Synpr",
        "Syt2",
        "Syt4",
        "Sytl4",
        "Th",
        "Tiparp",
        "Tmem108",
        "Traf4",
        "Ttn",
        "Ttyh2",
    ]
)
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


with open(genetypes_location, "rb") as f:
    genetypes = pickle.load(f)


# onehot encode cell classes
def oh_encode(lst):
    lst = np.array(lst)
    group_names = np.unique(lst)
    group_indexes = np.zeros((len(lst), len(group_names)), dtype=bool)
    for i, nm in enumerate(group_names):
        group_indexes[lst == nm, i] = True
    return group_names, group_indexes


cell_classes, cell_class_onehots = oh_encode(ad.obs["Cell_class"])


# In[39]:


# a function to construct a prediction problem for a subset of cells


def construct_problem(
    mask, target_gene, neighbor_genes, self_genes, filter_excitatory=False
):
    """
    mask -- set of cells
    target_gene -- gene to predict
    neighbor_genes -- names of genes which will be read from neighbors
    self_genes -- names of genes which will be read from target cell
    """

    feature_names = []

    # load subset of data relevant to mask
    local_processed_expression = np.log1p(
        ad.X[mask].astype(float)
    )  # get expression on subset of cells
    local_edges = connectivity_matrix[mask][:, mask]  # get edges for subset

    selfset_idxs = [
        gene_lookup[x] for x in self_genes
    ]  # collect the column indexes associated with them
    selfset_exprs = local_processed_expression[
        :, selfset_idxs
    ]  # collect ligand and receptor expressions

    feature_names += [x for x in self_genes]

    neighborset_idxs = [
        gene_lookup[x] for x in neighbor_genes
    ]  # collect the column indexes associated with them
    neighset_exprs = local_processed_expression[
        :, neighborset_idxs
    ]  # collect ligand and receptor expressions

    feature_names += [x + " from Neighbors" for x in neighbor_genes]

    n_neighs = local_edges @ np.ones(local_edges.shape[0])
    # print(local_edges)
    # print(n_neighs)
    neigh_avgs = (local_edges @ neighset_exprs) / n_neighs[
        :, None
    ]  # average ligand/receptor for neighbors

    neigh_cellclass_avgs = (local_edges @ cell_class_onehots[mask]) / n_neighs[
        :, None
    ]  # celltype simplex

    feature_names += [f"Cell Class {cell_types[x]}" for x in range(16)]

    positions = np.array(ad.obs[["Centroid_X", "Centroid_Y", "Bregma"]])[
        mask
    ]  # get positions

    feature_names += ["Centroid_X", "Centroid_Y", "Bregma"]

    covariates = np.c_[
        selfset_exprs, neigh_avgs, neigh_cellclass_avgs, positions
    ]  # collect all covariates
    predict = local_processed_expression[
        :, gene_lookup[target_gene]
    ]  # collect what we're supposed to predict

    # print(selfset_exprs.shape, neigh_avgs.shape, neigh_cellclass_avgs.shape, positions.shape)

    if filter_excitatory:

        excites = (ad.obs["Cell_class"] == "Excitatory")[
            mask
        ]  # get the subset of these cells which are excitatory
        covariates = covariates[excites]  # subset to excites
        predict = predict[excites]  # subset to excites

    return covariates, predict, feature_names


# # 0 vs. 60 LightGBM Test

# ### 0 Radius Graph

# In[41]:

ad = anndata.read_h5ad(h5ad_location)
row = np.zeros(0, dtype=int)
col = np.zeros(0, dtype=int)
nneigh = 10
radius = 0
mode = "rad"

ad = anndata.read_h5ad(h5ad_location)
if mode == "neighbors":
    connectivity_matrix = anndata.read_h5ad(
        connectivity_matrix_template % (nneigh, mode)
    ).X
if mode == "rad":
    connectivity_matrix = anndata.read_h5ad(
        connectivity_matrix_template % (radius, mode)
    ).X
gene_lookup = {x: i for (i, x) in enumerate(ad.var.index)}

for tid in tqdm.notebook.tqdm(np.unique(ad.obs["Tissue_ID"])):
    good = ad.obs["Tissue_ID"] == tid
    pos = np.array(ad.obs[good][["Centroid_X", "Centroid_Y"]])
    idxs = np.where(good)[0]
    if mode == "neighbors":
        if nneigh == 0:
            E = csr_matrix(np.eye(pos.shape[0]))
        else:
            p = sklearn.neighbors.BallTree(pos)
            E = sklearn.neighbors.kneighbors_graph(pos, nneigh, mode="connectivity")
        col = np.r_[col, idxs[E.tocoo().col]]
        row = np.r_[row, idxs[E.tocoo().row]]
    if mode == "rad":
        p = sp.spatial.cKDTree(pos)
        # E=p.query_ball_point(pos, r=radius, return_sorted=False)
        edges = p.query_pairs(r=radius)
        col = np.r_[
            col,
            np.concatenate(
                (idxs[[y for (x, y) in edges]], idxs[[x for (x, y) in edges]])
            ),
        ]
        row = np.r_[
            row,
            np.concatenate(
                (idxs[[x for (x, y) in edges]], idxs[[y for (x, y) in edges]])
            ),
        ]

E = (
    scipy.sparse.diags([1] * len(ad), 0)
    + sp.sparse.coo_matrix((np.ones(len(col)), (row, col)), shape=(len(ad), len(ad)))
).tocsr()

if mode == "neighbors":
    anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (nneigh, mode))
if mode == "rad":
    anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (radius, mode))


# In[77]:


results_Jackson_0 = {}
results_Roman_0 = {}


# In[79]:


neighset = genetypes["ligands"]
oset = np.r_[genetypes["ligands"], genetypes["receptors"]]
# oset=neighset

# oset=[]
# neighset=[]

for response_gene in tqdm.notebook.tqdm(response_genes):

    trainX, trainY, feature_names = construct_problem(
        (ad.obs["Animal_ID"] <= 30), response_gene, neighset, oset
    )
    testX, testY, feature_names = construct_problem(
        (ad.obs["Animal_ID"] > 30), response_gene, neighset, oset
    )

    mu = np.mean(trainX, axis=0)
    sig = np.std(trainX, axis=0)
    trainX_Jackson = (trainX - mu) / sig
    testX_Jackson = (testX - mu) / sig

    scaler = StandardScaler().fit(trainX)
    trainX_Roman = scaler.transform(trainX)
    testX_Roman = scaler.transform(testX)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        min_samples_leaf=2,
        verbose=1,
        random_state=129,
        max_iter=1000,
        n_iter_no_change=25,
    )
    model.fit(trainX_Roman, trainY)
    results_Roman_0[response_gene] = np.mean(np.abs(model.predict(testX_Roman) - testY))

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        min_samples_leaf=2,
        verbose=1,
        random_state=129,
        max_iter=1000,
        n_iter_no_change=25,
    )
    model.fit(trainX_Jackson, trainY)
    results_Jackson_0[response_gene] = np.mean(
        np.abs(model.predict(testX_Jackson) - testY)
    )


# ### 60 Radius Graph

# In[ ]:


ad = anndata.read_h5ad(h5ad_location)
row = np.zeros(0, dtype=int)
col = np.zeros(0, dtype=int)
nneigh = 10
radius = 60
mode = "rad"

ad = anndata.read_h5ad(h5ad_location)
if mode == "neighbors":
    connectivity_matrix = anndata.read_h5ad(
        connectivity_matrix_template % (nneigh, mode)
    ).X
if mode == "rad":
    connectivity_matrix = anndata.read_h5ad(
        connectivity_matrix_template % (radius, mode)
    ).X
gene_lookup = {x: i for (i, x) in enumerate(ad.var.index)}

for tid in tqdm.notebook.tqdm(np.unique(ad.obs["Tissue_ID"])):
    good = ad.obs["Tissue_ID"] == tid
    pos = np.array(ad.obs[good][["Centroid_X", "Centroid_Y"]])
    idxs = np.where(good)[0]
    if mode == "neighbors":
        if nneigh == 0:
            E = csr_matrix(np.eye(pos.shape[0]))
        else:
            p = sklearn.neighbors.BallTree(pos)
            E = sklearn.neighbors.kneighbors_graph(pos, nneigh, mode="connectivity")
        col = np.r_[col, idxs[E.tocoo().col]]
        row = np.r_[row, idxs[E.tocoo().row]]
    if mode == "rad":
        p = sp.spatial.cKDTree(pos)
        # E=p.query_ball_point(pos, r=radius, return_sorted=False)
        edges = p.query_pairs(r=radius)
        col = np.r_[
            col,
            np.concatenate(
                (idxs[[y for (x, y) in edges]], idxs[[x for (x, y) in edges]])
            ),
        ]
        row = np.r_[
            row,
            np.concatenate(
                (idxs[[x for (x, y) in edges]], idxs[[y for (x, y) in edges]])
            ),
        ]

E = (
    scipy.sparse.diags([1] * len(ad), 0)
    + sp.sparse.coo_matrix((np.ones(len(col)), (row, col)), shape=(len(ad), len(ad)))
).tocsr()

if mode == "neighbors":
    anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (nneigh, mode))
if mode == "rad":
    anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (radius, mode))

with open("Jackson0.json", "w") as fp:
    json.dump(results_Jackson_0, fp)

with open("Roman0.json", "w") as fp:
    json.dump(results_Roman_0, fp)

# In[ ]:

results_Jackson_60 = {}
results_Roman_60 = {}


# In[ ]:


neighset = genetypes["ligands"]
oset = np.r_[genetypes["ligands"], genetypes["receptors"]]
# oset=neighset

# oset=[]
# neighset=[]

for response_gene in tqdm.notebook.tqdm(response_genes):

    trainX, trainY, feature_names = construct_problem(
        (ad.obs["Animal_ID"] <= 30), response_gene, neighset, oset
    )
    testX, testY, feature_names = construct_problem(
        (ad.obs["Animal_ID"] > 30), response_gene, neighset, oset
    )

    mu = np.mean(trainX, axis=0)
    sig = np.std(trainX, axis=0)
    trainX_Jackson = (trainX - mu) / sig
    testX_Jackson = (testX - mu) / sig

    scaler = StandardScaler().fit(trainX)
    trainX_Roman = scaler.transform(trainX)
    testX_Roman = scaler.transform(testX)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        min_samples_leaf=2,
        verbose=1,
        random_state=129,
        max_iter=1000,
        n_iter_no_change=25,
    )
    model.fit(trainX_Roman, trainY)
    results_Roman_60[response_gene] = np.mean(
        np.abs(model.predict(testX_Roman) - testY)
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        min_samples_leaf=2,
        verbose=1,
        random_state=129,
        max_iter=1000,
        n_iter_no_change=25,
    )
    model.fit(trainX_Jackson, trainY)
    results_Jackson_60[response_gene] = np.mean(
        np.abs(model.predict(testX_Jackson) - testY)
    )


# In[ ]:


# write 0 vs. 60 stuff here

with open("Jackson60.json", "w") as fp:
    json.dump(results_Jackson_60, fp)

with open("Roman60.json", "w") as fp:
    json.dump(results_Roman_60, fp)
