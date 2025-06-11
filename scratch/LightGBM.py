# pylint: disable=invalid-name, too-many-branches, too-many-statements, unspecified-encoding, R0801
import os
import pickle
import json
import shap
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import tqdm.notebook
import sklearn.neighbors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

FILENAME = "LightGBM_results_NO_celltype_WITHRECEPTORS.json"

rng = np.random.default_rng()

original_url = "https://datadryad.org/stash/downloads/file_stream/67671"
csv_location = "../data/raw/merfish.csv"
h5ad_location = (
    "/nfs/turbo/lsa-regier/scratch/roko/data/spatial/moffit_merfish/original_file.h5ad"
)

connectivity_matrix_template = (
    "/nfs/turbo/lsa-regier/scratch/roko/data/spatial/moffit_merfish/"
    "connectivity_%d%s.h5ad"
)
genetypes_location = (
    "/nfs/turbo/lsa-regier/scratch/roko/data/spatial/moffit_merfish/genetypes.pkl"
)

dataframe = pd.read_csv(csv_location)

dct = {}
for colnm, dtype in zip(dataframe.keys()[:9], dataframe.dtypes[:9]):
    if dtype.kind == "O":
        dct[colnm] = np.require(dataframe[colnm], dtype="U36")
    else:
        dct[colnm] = np.require(dataframe[colnm])
# change expression here to make it synthetic
expression = np.array(dataframe[dataframe.keys()[9:]]).astype(np.float64)
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

# Check if the file exists, if not, create it
if not os.path.exists(genetypes_location):
    with open(genetypes_location, "wb") as f:
        pickle.dump({}, f)  # Initialize an empty dictionary if the file doesn't exist

# Load the file
with open(genetypes_location, "rb") as f:
    genetypes = pickle.load(f)

# neighset = genetypes["ligands"]
neighset = np.r_[genetypes["ligands"], genetypes["receptors"]]
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
    include_celltypes=True,
):
    """
    mask -- set of cells
    target_gene -- gene to predict
    neighbor_genes -- names of genes which will be read from neighbors
    self_genes -- names of genes which will be read from target cell
    """

    collected_feature_names = []

    # load subset of data relevant to mask
    local_edges = connectivity_matrix[mask][:, mask]  # get edges for subset
    print(np.sum(local_edges, axis=0).max())
    selfset_idxs = [
        gene_lookup[x] for x in self_genes
    ]  # collect the column indexes associated with them

    # with h5py.File("../data/raw/merfish.hdf5", "r") as data:
    #     local_processed_expression = data["expression"][:][mask].astype("float64")
    # load subset of data relevant to mask
    local_processed_expression = ad.X.astype(float)[
        mask
    ]  # get expression on subset of cells
    selfset_exprs = local_processed_expression[
        :, selfset_idxs
    ]  # collect ligand and receptor expressions

    # perform the log transform
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
    # DIVISION BY NEIGHBORS GIVES AVERAGE. OTHERWISE THIS IS A SUM.
    neigh_avgs = (
        local_edges @ neighset_exprs / n_neighs[:, None]
    )  # average ligand/receptor for neighbors

    positions = np.array(ad.obs[["Centroid_X", "Centroid_Y", "Bregma"]])[
        mask
    ]  # get positions

    collected_feature_names += ["Centroid_X", "Centroid_Y", "Bregma"]

    if include_celltypes:
        neigh_cellclass_avgs = (local_edges @ cell_class_onehots[mask]) / n_neighs[
            :, None
        ]  # celltype simplex

        collected_feature_names += [
            f"Self Cell Class {cell_types[x]}" for x in range(16)
        ]

        collected_feature_names += [f"Cell Class {cell_types[x]}" for x in range(16)]

    covariates = np.c_[selfset_exprs, neigh_avgs, positions]  # collect all covariates

    if include_celltypes:
        covariates = np.c_[covariates, neigh_cellclass_avgs, cell_class_onehots[mask]]

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

if os.path.exists(FILENAME):
    with open(FILENAME, "r") as results:
        try:
            results_dict = json.load(results)
        except json.decoder.JSONDecodeError:
            results_dict = {}
else:
    results_dict = {}

SHAP = False

for radius in range(5, 65, 5):

    # Build the Current Radius Graph
    ad = anndata.read_h5ad(h5ad_location)
    row = np.zeros(0, dtype=int)
    col = np.zeros(0, dtype=int)

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
        col = np.r_[col, idxs[[y for (x, y) in edges]]]
        row = np.r_[row, idxs[[x for (x, y) in edges]]]

    E = (
        sp.sparse.coo_matrix((np.ones(len(col)), (row, col)), shape=(len(ad), len(ad)))
    ).tocsr()

    anndata.AnnData(E).write_h5ad(connectivity_matrix_template % (radius, "rad"))

    connectivity_matrix = anndata.read_h5ad(
        connectivity_matrix_template % (radius, "rad")
    ).X
    gene_lookup = {x: i for (i, x) in enumerate(ad.var.index)}

    for response_gene in tqdm.notebook.tqdm(response_genes):

        trainX, trainY, feature_names = construct_problem(
            (ad.obs["Animal_ID"] <= 30),
            response_gene,
            neighset,
            oset,
            include_celltypes=False,
        )
        testX, testY, feature_names = construct_problem(
            (ad.obs["Animal_ID"] > 30),
            response_gene,
            neighset,
            oset,
            include_celltypes=False,
        )
        print(trainX.shape, trainY.shape)
        print(testX.shape, testY.shape)

        df = pd.DataFrame(testX, columns=feature_names)

        scaler = StandardScaler().fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            min_samples_leaf=2,
            verbose=0,
            random_state=129,
            max_iter=1000,
            n_iter_no_change=10,
            tol=0.00001,
        )

        model.fit(trainX, trainY)
        results_dict[f"LightGBM {radius}"] = results_dict.get(f"LightGBM {radius}", {})
        results_dict[f"LightGBM {radius}"][response_gene] = np.mean(
            (model.predict(testX) - testY) ** 2
        )

        if SHAP:
            shap_values = shap.TreeExplainer(model).shap_values(df)
            shap.summary_plot(shap_values, df, show=False)
            plt.title(f"(LightGBM, {radius})")
            plt.rcParams["figure.dpi"] = 100
            plt.rcParams["savefig.dpi"] = 100
            plt.savefig(
                f"static/shap/(LightGBM, {radius}).png",
                bbox_inches="tight",
            )
            plt.clf()

        model = sklearn.linear_model.LinearRegression()
        model.fit(trainX, trainY)
        results_dict[f"OLS {radius}"] = results_dict.get(f"OLS {radius}", {})
        results_dict[f"OLS {radius}"][response_gene] = np.mean(
            (model.predict(testX) - testY) ** 2
        )
        if SHAP:
            shap_values = shap.LinearExplainer(model, trainX).shap_values(df)
            shap.summary_plot(shap_values, df, show=False)
            plt.title(f"(LightGBM, {radius})")
            plt.rcParams["figure.dpi"] = 100
            plt.rcParams["savefig.dpi"] = 100
            plt.savefig(
                f"static/shap/(OLS, {radius}).png",
                bbox_inches="tight",
            )
            plt.clf()

        model = sklearn.linear_model.Ridge()
        tuned_parameters = {"alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]}
        grid_search = GridSearchCV(
            model, tuned_parameters, scoring="neg_mean_squared_error", cv=5
        )
        grid_search.fit(trainX, trainY)
        results_dict[f"Ridge {radius}"] = results_dict.get(f"Ridge {radius}", {})
        results_dict[f"Ridge {radius}"][response_gene] = np.mean(
            (grid_search.predict(testX) - testY) ** 2
        )
        if SHAP:
            shap_values = shap.LinearExplainer(
                grid_search.best_estimator_, trainX
            ).shap_values(df)
            shap.summary_plot(shap_values, df, show=False)
            plt.title(f"(Ridge, {radius})")
            plt.rcParams["figure.dpi"] = 100
            plt.rcParams["savefig.dpi"] = 100
            plt.savefig(
                f"static/shap/(Ridge, {radius}).png",
                bbox_inches="tight",
            )
            plt.clf()

        model = sklearn.linear_model.Lasso(alpha=0.1)
        tuned_parameters = {"alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]}
        grid_search = GridSearchCV(
            model, tuned_parameters, scoring="neg_mean_squared_error", cv=5
        )
        grid_search.fit(trainX, trainY)
        results_dict[f"Lasso {radius}"] = results_dict.get(f"Lasso {radius}", {})
        results_dict[f"Lasso {radius}"][response_gene] = np.mean(
            (grid_search.predict(testX) - testY) ** 2
        )
        if SHAP:
            shap_values = shap.LinearExplainer(
                grid_search.best_estimator_, trainX
            ).shap_values(df)
            shap.summary_plot(shap_values, df, show=False)
            plt.title(f"(Lasso, {radius})")
            plt.rcParams["figure.dpi"] = 100
            plt.rcParams["savefig.dpi"] = 100
            plt.savefig(
                f"static/shap/(Lasso, {radius}).png",
                bbox_inches="tight",
            )
            plt.clf()

        # model = sklearn.linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5)
        # tuned_parameters = {
        #     "alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        #     "l1_ratio": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99],
        # }
        # grid_search = GridSearchCV(
        #     model, tuned_parameters, scoring="neg_mean_squared_error", cv=5
        # )
        # grid_search.fit(trainX, trainY)
        # results_dict[f"ElasticNet {radius}"] =
        # results_dict.get(f"ElasticNet {radius}", 0)
        # results_dict[f"ElasticNet {radius}"] += np.mean(
        #     (grid_search.predict(testX) - testY) ** 2
        # )
        # if SHAP:
        #     shap_values = shap.LinearExplainer(
        #         grid_search.best_estimator_, trainX
        #     ).shap_values(df)
        #     shap.summary_plot(shap_values, df, show=False)
        #     plt.title(f"(ElasticNet, {radius})")
        #     plt.rcParams["figure.dpi"] = 100
        #     plt.rcParams["savefig.dpi"] = 100
        #     plt.savefig(
        #         f"static/shap/(ElasticNet, {radius}).png",
        #         bbox_inches="tight",
        #     )
        #     plt.clf()

        with open(FILENAME, "w") as results:
            json.dump(results_dict, results)
