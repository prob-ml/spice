# pylint: disable=invalid-name, too-many-branches, too-many-statements, unspecified-encoding, R0801
import pickle
import json
import shap
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import torch
import tqdm.notebook
import sklearn.neighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng()


original_url = "https://datadryad.org/stash/downloads/file_stream/67671"
csv_location = "data/spatial/moffit_merfish/original_file.csv"
h5ad_location = "data/spatial/moffit_merfish/original_file.h5ad"
connectivity_matrix_template = "data/spatial/moffit_merfish/connectivity_%d%s.h5ad"
genetypes_location = "data/spatial/moffit_merfish/genetypes.pkl"

dataframe = pd.read_csv(csv_location)

dct = {}
for colnm, dtype in zip(dataframe.keys()[:9], dataframe.dtypes[:9]):
    if dtype.kind == "O":
        dct[colnm] = np.require(dataframe[colnm], dtype="U36")
    else:
        dct[colnm] = np.require(dataframe[colnm])
# change expression here to make it synthetic
expression = np.array(dataframe[dataframe.keys()[9:]]).astype(np.float16)
print(expression)
gene_names = np.array(dataframe.keys()[9:], dtype="U80")
cellid = dct.pop("Cell_ID")

ad = anndata.AnnData(
    X=expression,
    var=pd.DataFrame(index=gene_names),
    obs=pd.DataFrame(dct, index=cellid),
)

ad.write_h5ad(h5ad_location)

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

neighset = genetypes["ligands"]
oset = np.r_[genetypes["ligands"], genetypes["receptors"]]
# oset=neighset

# onehot encode cell classes
def oh_encode(lst):
    lst = np.array(lst)
    group_names = np.unique(lst)
    group_indexes = np.zeros((len(lst), len(group_names)), dtype=bool)
    for i, nm in enumerate(group_names):
        group_indexes[lst == nm, i] = True
    return group_names, group_indexes


cell_classes, cell_class_onehots = oh_encode(ad.obs["Cell_class"])

# a function to construct a prediction problem for a subset of cells


def construct_problem(
    mask,
    target_gene,
    neighbor_genes,
    self_genes,
    filter_excitatory=False,
    synthetic_mode=0,
):
    """
    mask -- set of cells
    target_gene -- gene to predict
    neighbor_genes -- names of genes which will be read from neighbors
    self_genes -- names of genes which will be read from target cell
    """

    collected_feature_names = []

    # load subset of data relevant to mask
    local_processed_expression = np.log1p(
        ad.X[mask].astype(float)
    )  # get expression on subset of cells
    local_edges = connectivity_matrix[mask][:, mask]  # get edges for subset

    true_local_edges = true_connectivity_matrix[mask][:, mask]  # get edges for subset

    selfset_idxs = [
        gene_lookup[x] for x in self_genes
    ]  # collect the column indexes associated with them

    if synthetic_mode == 0:
        cell_volume = 5
        local_processed_expression = (
            torch.distributions.negative_binomial.NegativeBinomial(1, 0.5).sample(
                local_processed_expression.shape
            )
            / cell_volume
        ).numpy()
        selfset_exprs = local_processed_expression[
            :, selfset_idxs
        ]  # collect ligand and receptor expressions
        for i in range(local_processed_expression.shape[0]):
            neighboring_gene1_expr = np.sum(
                selfset_exprs[list(true_local_edges[i].nonzero()[1]), 1]
            )
            if neighboring_gene1_expr > 1:
                local_processed_expression[i, 0] = neighboring_gene1_expr
            else:
                local_processed_expression[i, 0] = 0

    if synthetic_mode == 1:
        local_processed_expression = (
            torch.distributions.exponential.Exponential(10)
            .rsample(local_processed_expression.shape)
            .numpy()
        )
        selfset_exprs = local_processed_expression[
            :, selfset_idxs
        ]  # collect ligand and receptor expressions
        for i in range(local_processed_expression.shape[0]):
            neighboring_gene1_expr = np.sum(
                selfset_exprs[list(true_local_edges[i].nonzero()[1]), 1]
            )
            if neighboring_gene1_expr > 1:
                local_processed_expression[i, 0] += neighboring_gene1_expr

    elif synthetic_mode == 2:
        local_processed_expression = torch.exp(
            torch.distributions.normal.Normal(0, 1).rsample(
                local_processed_expression.shape
            )
        ).numpy()
        selfset_exprs = local_processed_expression[
            :, selfset_idxs
        ]  # collect ligand and receptor expressions
        for i in range(local_processed_expression.shape[0]):
            neighboring_gene_expr = np.mean(
                selfset_exprs[list(true_local_edges[i].nonzero()[1]), 1:10]
            )
            local_processed_expression[i, 0] += neighboring_gene_expr * (
                2 ** np.sign(neighboring_gene_expr - 1.6)
            )

    elif synthetic_mode == 3:
        local_processed_expression = torch.exp(
            torch.distributions.normal.Normal(0, 0.25).rsample(
                local_processed_expression.shape
            )
        ).numpy()
        selfset_exprs = local_processed_expression[
            :, selfset_idxs
        ]  # collect ligand and receptor expressions
        average_gene1_expr = np.mean(selfset_exprs[:, 1])
        average_gene2_expr = np.mean(selfset_exprs[:, 2])
        for i in range(local_processed_expression.shape[0]):
            neighboring_gene1_expr = np.mean(
                selfset_exprs[list(true_local_edges[i].nonzero()[1]), 1]
            )
            neighboring_gene2_expr = np.mean(
                selfset_exprs[list(true_local_edges[i].nonzero()[1]), 2]
            )
            if (
                np.sign(neighboring_gene1_expr - average_gene1_expr)
                - np.sign(neighboring_gene2_expr - average_gene2_expr)
            ) > 0:
                local_processed_expression[i, 0] *= 2
            else:
                local_processed_expression[i, 0] *= 0.5

    elif synthetic_mode == 4:
        local_processed_expression = (
            torch.distributions.exponential.Exponential(10)
            .rsample(local_processed_expression.shape)
            .numpy()
        )
        selfset_exprs = local_processed_expression[:, selfset_idxs]
        for i in range(local_processed_expression.shape[0]):
            neighboring_gene1_expr = selfset_exprs[
                true_edges.T[:, true_edges.T[0] == i][1], 1
            ]
            neighboring_gene1_distances = true_distances.T[1][true_distances.T[0] == i]
            neighboring_gene1_expr = np.dot(
                neighboring_gene1_expr, 1 / (1 + neighboring_gene1_distances)
            )
            local_processed_expression[i, 0] = neighboring_gene1_expr

    elif synthetic_mode == 5:
        local_processed_expression = (
            torch.distributions.exponential.Exponential(10)
            .rsample(local_processed_expression.shape)
            .numpy()
        )
        selfset_exprs = local_processed_expression[:, selfset_idxs]

        def __weight(x):
            return 1 - np.arcsinh(5.863 * x) / 5.863

        for i in range(local_processed_expression.shape[0]):
            neighboring_gene1_expr = selfset_exprs[
                true_edges.T[:, true_edges.T[0] == i][1], 1
            ]
            neighboring_gene1_distances = true_distances.T[1][true_distances.T[0] == i]
            neighboring_gene1_expr = np.dot(
                neighboring_gene1_expr, __weight(neighboring_gene1_distances)
            )
            local_processed_expression[i, 0] = neighboring_gene1_expr

    local_processed_expression = np.log1p(local_processed_expression)
    selfset_exprs = np.log1p(selfset_exprs)
    collected_feature_names += list(self_genes)

    neighborset_idxs = [
        gene_lookup[x] for x in neighbor_genes
    ]  # collect the column indexes associated with them
    neighset_exprs = local_processed_expression[
        :, neighborset_idxs
    ]  # collect ligand and receptor expressions

    collected_feature_names += [x + " from Neighbors" for x in neighbor_genes]

    n_neighs = local_edges @ np.ones(local_edges.shape[0])
    # print(local_edges)
    # print(n_neighs)
    neigh_avgs = (local_edges @ neighset_exprs) / n_neighs[
        :, None
    ]  # average ligand/receptor for neighbors

    neigh_cellclass_avgs = (local_edges @ cell_class_onehots[mask]) / n_neighs[
        :, None
    ]  # celltype simplex

    collected_feature_names += [f"Cell Class {cell_types[x]}" for x in range(16)]

    positions = np.array(ad.obs[["Centroid_X", "Centroid_Y", "Bregma"]])[
        mask
    ]  # get positions

    collected_feature_names += ["Centroid_X", "Centroid_Y", "Bregma"]

    covariates = np.c_[
        selfset_exprs, neigh_avgs, neigh_cellclass_avgs, positions
    ]  # collect all covariates
    predict = local_processed_expression[
        :, gene_lookup[target_gene]
    ]  # collect what we're supposed to predict

    if filter_excitatory:

        excites = (ad.obs["Cell_class"] == "Excitatory")[
            mask
        ]  # get the subset of these cells which are excitatory
        covariates = covariates[excites]  # subset to excites
        predict = predict[excites]  # subset to excites

    return covariates, predict, collected_feature_names


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

# Building Radius 30 Graph (Ground Truth)
ad = anndata.read_h5ad(h5ad_location)
row = np.zeros(0, dtype=int)
col = np.zeros(0, dtype=int)
true_radius = 30

for tid in tqdm.notebook.tqdm(np.unique(ad.obs["Tissue_ID"])):
    good = ad.obs["Tissue_ID"] == tid
    pos = np.array(ad.obs[good][["Centroid_X", "Centroid_Y"]])
    idxs = np.where(good)[0]
    p = sklearn.neighbors.KDTree(pos)
    true_edges, true_distances = p.query_radius(
        pos, r=true_radius, return_distance=True
    )
    true_distances = np.concatenate(
        [
            np.c_[
                np.repeat(i, len(true_distances[i])),
                list(true_distances[i]),
            ]
            for i in range(len(true_distances))
        ],
        axis=0,
    )
    true_edges = np.concatenate(
        [
            np.c_[
                np.repeat(i, len(true_edges[i])),
                list(true_edges[i]),
            ]
            for i in range(len(true_edges))
        ],
        axis=0,
    )
    col = np.r_[
        col,
        np.concatenate(
            (idxs[[y for (x, y) in true_edges]], idxs[[x for (x, y) in true_edges]])
        ),
    ]
    row = np.r_[
        row,
        np.concatenate(
            (idxs[[x for (x, y) in true_edges]], idxs[[y for (x, y) in true_edges]])
        ),
    ]

E = (
    scipy.sparse.diags([1] * len(ad), 0)
    + sp.sparse.coo_matrix((np.ones(len(col)), (row, col)), shape=(len(ad), len(ad)))
).tocsr()

anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (true_radius, "rad"))

# load data
# These are set above. You can change these here if you want though.
# radius=60
# mode="rad"
ad = anndata.read_h5ad(h5ad_location)
true_connectivity_matrix = anndata.read_h5ad(
    connectivity_matrix_template % (true_radius, "rad")
).X
gene_lookup = {x: i for (i, x) in enumerate(ad.var.index)}

results_dict = {}

response_gene = "Ace2"  # make the response gene the first response

for synth_experiment in range(6):
    for radius_value in range(0, 65, 5):

        # Build the Current Radius Graph
        ad = anndata.read_h5ad(h5ad_location)
        row = np.zeros(0, dtype=int)
        col = np.zeros(0, dtype=int)
        radius = radius_value

        for tid in tqdm.notebook.tqdm(np.unique(ad.obs["Tissue_ID"])):
            good = ad.obs["Tissue_ID"] == tid
            pos = np.array(ad.obs[good][["Centroid_X", "Centroid_Y"]])
            idxs = np.where(good)[0]
            p = sklearn.neighbors.KDTree(pos)
            # E=p.query_ball_point(pos, r=radius, return_sorted=False)
            edges, distances = p.query_radius(pos, r=radius, return_distance=True)
            distances = np.concatenate(
                [
                    np.c_[
                        np.repeat(i, len(distances[i])),
                        list(distances[i]),
                    ]
                    for i in range(len(distances))
                ],
                axis=0,
            )
            edges = np.concatenate(
                [
                    np.c_[
                        np.repeat(i, len(edges[i])),
                        list(edges[i]),
                    ]
                    for i in range(len(edges))
                ],
                axis=0,
            )
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
            + sp.sparse.coo_matrix(
                (np.ones(len(col)), (row, col)), shape=(len(ad), len(ad))
            )
        ).tocsr()

        anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (radius, "rad"))

        ad = anndata.read_h5ad(h5ad_location)
        connectivity_matrix = anndata.read_h5ad(
            connectivity_matrix_template % (radius, "rad")
        ).X
        gene_lookup = {x: i for (i, x) in enumerate(ad.var.index)}

        trainX, trainY, feature_names = construct_problem(
            (ad.obs["Animal_ID"] <= 30),
            response_gene,
            neighset,
            oset,
            synthetic_mode=synth_experiment,
        )
        testX, testY, feature_names = construct_problem(
            (ad.obs["Animal_ID"] > 30),
            response_gene,
            neighset,
            oset,
            synthetic_mode=synth_experiment,
        )

        scaler = StandardScaler().fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            min_samples_leaf=2,
            verbose=1,
            random_state=129,
            max_iter=1000,
            n_iter_no_change=10,
            tol=0.001,
        )
        model.fit(trainX, trainY)
        results_dict[f"LightGBM {radius_value} {synth_experiment}"] = np.mean(
            np.mean((model.predict(testX) - testY) ** 2)
        )

        df = pd.DataFrame(testX, columns=feature_names)
        shap_values = shap.TreeExplainer(model).shap_values(df)
        shap.summary_plot(shap_values, df, show=False)
        plt.title(f"(LightGBM, {radius_value}, {synth_experiment})")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100
        plt.savefig(
            f"static/shap_recovery/(LightGBM, {radius_value}, {synth_experiment}).png",
            bbox_inches="tight",
        )
        plt.clf()

        model = sklearn.linear_model.Ridge(alpha=1.0)
        model.fit(trainX, trainY)
        results_dict[f"Ridge {radius_value} {synth_experiment}"] = np.mean(
            np.mean((model.predict(testX) - testY) ** 2)
        )

        df = pd.DataFrame(testX, columns=feature_names)
        shap_values = shap.LinearExplainer(model, trainX).shap_values(df)
        shap.summary_plot(shap_values, df, show=False)
        plt.title(f"(Ridge, {radius_value}, {synth_experiment})")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100
        plt.savefig(
            f"static/shap_recovery/(Ridge, {radius_value}, {synth_experiment}).png",
            bbox_inches="tight",
        )
        plt.clf()

        model = sklearn.linear_model.Lasso(alpha=1.0)
        model.fit(trainX, trainY)
        results_dict[f"Lasso {radius_value} {synth_experiment}"] = np.mean(
            np.mean((model.predict(testX) - testY) ** 2)
        )

        df = pd.DataFrame(testX, columns=feature_names)
        shap_values = shap.LinearExplainer(model, trainX).shap_values(df)
        shap.summary_plot(shap_values, df, show=False)
        plt.title(f"(Lasso, {radius_value}, {synth_experiment})")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100
        plt.savefig(
            f"static/shap_recovery/(Lasso, {radius_value}, {synth_experiment}).png",
            bbox_inches="tight",
        )
        plt.clf()

        model = sklearn.linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
        model.fit(trainX, trainY)
        results_dict[f"ElasticNet {radius_value} {synth_experiment}"] = np.mean(
            np.mean((model.predict(testX) - testY) ** 2)
        )

        df = pd.DataFrame(testX, columns=feature_names)
        shap_values = shap.LinearExplainer(model, trainX).shap_values(df)
        shap.summary_plot(shap_values, df, show=False)
        plt.title(f"(ElasticNet, {radius_value}, {synth_experiment})")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100
        plt.savefig(
            f"static/shap/(ElasticNet,{radius_value},{synth_experiment}).png",
            bbox_inches="tight",
        )
        plt.clf()

    with open("LightGBM_synthetic_results.json", "w") as synth_results:
        json.dump(results_dict, synth_results)
