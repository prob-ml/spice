import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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
N, P = data.shape

# Define the radius values for the r-ball.
r_values = range(0, 41, 5)
mse_performances = {}
training_mse = {}
testing_mse = {}

warnings.filterwarnings("ignore")

# Define parameter grid for ElasticNet
alphas = [0.1, 1.0, 10.0, 100.0]
l1_ratios = [0.1, 0.5, 0.7, 0.9, 1.0]

# Iterate over radius values
for r in tqdm(r_values, desc="Radius values"):
    training_mse[r] = {}
    testing_mse[r] = {}
    nn = NearestNeighbors(radius=r)
    nn.fit(locations)
    indices = nn.radius_neighbors(locations, return_distance=False)

    neighbor_means = np.array(
        [
            data[inds].mean(axis=0) if len(inds) > 0 else np.full(P, np.nan)
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
        print(f"\n🧬 Processing gene {i}/{len(response_gene_names)}: {response_gene}")
        y = xenium_df_outputs[response_gene]

        X_train, X_test, y_train, y_test = train_test_split(
            combined_df, y, test_size=0.2, random_state=42
        )

        model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=5, random_state=42)
        model.fit(X_train, y_train)

        print(f"✅ Best alpha: {model.alpha_}, Best l1_ratio: {model.l1_ratio_}")

        y_pred_train = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        print(f"✅ Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        training_mse[r][response_gene] = train_mse
        testing_mse[r][response_gene] = test_mse

with open("xenium_elasticnet_train.json", "w", encoding="utf-8") as f:
    json.dump(training_mse, f)

with open("xenium_elasticnet_test.json", "w", encoding="utf-8") as f:
    json.dump(testing_mse, f)
