import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import lightgbm as lgb


def assign_regions(location_data, num_splits):
    """
    location_data: (N, 2) array of x,y
    n_splits: how many times to bisect (each time doubles # regions)
    returns region_ids: (N,) ints in [0, 2**n_splits)
    """
    num_rows, dims = location_data.shape  # dims == 2 here
    region_ids = np.zeros(num_rows, dtype=int)  # start with one region (id 0)
    n_regions = 1

    for _ in range(num_splits):
        for dim in range(dims):  #   first x (dim=0), then y (dim=1)
            new_ids = np.zeros_like(region_ids)  # ← moved **inside** dim-loop
            for rid in range(n_regions):  #   operate on *current* labels
                mask = region_ids == rid
                if not mask.any():  #   skip empty regions
                    continue
                median = np.median(location_data[mask, dim])
                left = mask & (location_data[:, dim] < median)
                right = mask & (location_data[:, dim] >= median)
                new_ids[left] = 2 * rid
                new_ids[right] = 2 * rid + 1
            region_ids = new_ids  # ← update **before** next axis-split
            n_regions *= 2  #   doubled once per axis
        print(f"We created {n_regions} regions")  # will show 4, 16, 64, …

    return region_ids


NON_RESPONSE_FILE = "../spatial/non_response_blank_removed_xenium.txt"
with open(NON_RESPONSE_FILE, "r", encoding="utf-8") as f:
    non_response_genes = f.read().split(",")

non_response_genes = [int(x) for x in non_response_genes]

xenium_df = pd.read_csv("../data/raw/xenium.csv", index_col="cell_id")

locations = xenium_df.iloc[:, -4:-1]
location_names = ["x_location", "y_location", "z_location", "qv"]

xenium_df = xenium_df.iloc[:, :-4]

non_response_gene_names = xenium_df.columns[non_response_genes]
response_gene_names = xenium_df.columns[
    ~xenium_df.columns.isin(non_response_gene_names | location_names)
]
print(non_response_gene_names, response_gene_names)

xenium_df_inputs = xenium_df.iloc[:, non_response_genes]

xenium_df_outputs = xenium_df.iloc[:, xenium_df.columns.isin(response_gene_names)]

# log1p Transforms
xenium_df_inputs = np.log1p(xenium_df_inputs)
xenium_df_outputs = np.log1p(xenium_df_outputs)

data = xenium_df_inputs.values
num_obs, num_features = data.shape

# Define the radius values for the r-ball.
r_values = range(0, 41, 5)
mse_performances = {}

warnings.filterwarnings("ignore")

# Define parameter for LightGBM
params = {
    "objective": "regression",
    "metric": "mse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "n_estimators": 500,
    "max_depth": 7,
    "min_data_in_leaf": 15,
}

n_splits = 3
n_folds = 4
# Iterate over radius values
# Pre-compute region assignments and masks for all folds
if n_splits > 0:
    locations_array = locations[["x_location", "y_location"]].values
    regs = assign_regions(locations_array, n_splits)
    n_graphs = len(np.unique(regs))
    unique_regs = np.sort(np.unique(regs))

    # Pre-compute masks for all folds
    fold_masks = []
    for fold in range(n_folds):
        test_regs = unique_regs[
            fold * n_graphs // n_folds : (fold + 1) * n_graphs // n_folds
        ]
        train_regs = np.concatenate(
            [
                unique_regs[: fold * n_graphs // n_folds],
                unique_regs[(fold + 1) * n_graphs // n_folds :],
            ]
        )
        train_mask = np.isin(regs, train_regs)
        test_mask = ~train_mask
        fold_masks.append((train_mask, test_mask))

for fold in range(n_folds):
    training_mse = {}
    testing_mse = {}

    for r in tqdm(r_values, desc="Radius values"):
        training_mse[r] = {}
        testing_mse[r] = {}
        nn = NearestNeighbors(radius=r)
        nn.fit(locations)
        indices = nn.radius_neighbors(locations, return_distance=False)

        neighbor_means = np.array(
            [
                data[inds].mean(axis=0)
                if len(inds) > 0
                else np.full(num_features, np.nan)
                for inds in indices
            ]
        )

        if np.isnan(neighbor_means).any():
            raise ValueError("neighbor_means contains NaN values.")

        combined = np.hstack([data, neighbor_means])
        combined_df = pd.DataFrame(
            combined,
            columns=list(xenium_df_inputs.columns)
            + [f"{col}_mean" for col in xenium_df_inputs.columns],
        )

        for i, response_gene in enumerate(response_gene_names, start=1):
            print(
                f"\n🚀 Processing gene {i}/{len(response_gene_names)}: {response_gene}"
            )
            y = xenium_df_outputs[response_gene]

            if n_splits > 0:
                train_mask, test_mask = fold_masks[fold]
                X_train_full = combined_df.iloc[train_mask]
                y_train_full = y.values[train_mask]
                X_test = combined_df.iloc[test_mask]
                y_test = y.values[test_mask]

                vcut = int(len(X_train_full) * 0.9)
                X_train = X_train_full.iloc[:vcut]
                y_train = y_train_full[:vcut]
                X_val = X_train_full.iloc[vcut:]
                y_val = y_train_full[vcut:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    combined_df, y, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=42
                )

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_val, label=y_val)
            dtest = lgb.Dataset(X_test, label=y_test)

            model = lgb.train(
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(stopping_rounds=30)],
                num_boost_round=params["n_estimators"],
            )

            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            mse = mean_squared_error(y_val, val_preds)
            print(f"✅ Validation MSE: {mse:.4f}")

            train_preds = model.predict(X_train, num_boost_round=model.best_iteration)
            train_mse = mean_squared_error(y_train, train_preds)
            print(f"✅ Train MSE: {train_mse:.4f}")

            test_preds = model.predict(X_test, num_boost_round=model.best_iteration)
            test_mse = mean_squared_error(y_test, test_preds)
            print(f"✅ Test MSE: {test_mse:.4f}")

            training_mse[r][response_gene] = train_mse
            testing_mse[r][response_gene] = test_mse

    with open(f"xenium_lightgbm_train_FOLD={fold}.json", "w", encoding="utf-8") as f:
        json.dump(training_mse, f)

    with open(f"xenium_lightgbm_test_FOLD={fold}.json", "w", encoding="utf-8") as f:
        json.dump(testing_mse, f)
