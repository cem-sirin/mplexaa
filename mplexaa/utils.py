import os
from itertools import product
import numpy as np
import pandas as pd

dpath = lambda dataset: f"datasets/{dataset}/Dataset"
listfiles = lambda dataset: [x for x in os.listdir(dpath(dataset)) if not x.startswith(".")]
datasets = [
    ("cns", ["calls", "fb_friends", "sms"]),
    ("csa", ["lunch", "facebook", "work"]),
    ("ckm", ["advice", "discussion", "friend"]),
    ("eua", ["Lufthansa", "Ryanair", "Easyjet"]),
    ("laz", ["advice", "friendship", "co-work"]),
    ("lon", ["tube", "overground", "light_railway"]),
    ("vic", ["get_on_with", "best_friends", "work_with"]),
]


def parse_files(dataset):
    files = listfiles(dataset)
    layers = [x for x in files if "layers" in x]
    edges = [x for x in files if "edges" in x]
    nodes = [x for x in files if "nodes" in x]

    file_dict = {}
    file_dict["layers"] = f"{dpath(dataset)}/{layers[0]}" if len(layers) > 0 else None
    file_dict["edges"] = f"{dpath(dataset)}/{edges[0]}" if len(edges) > 0 else None
    file_dict["nodes"] = f"{dpath(dataset)}/{nodes[0]}" if len(nodes) > 0 else None
    return file_dict


def get_datasets():
    return datasets


def get_dataset_names():
    return [x[0] for x in datasets]


### Functions to generate eta grid ###
def get_positive_eta_grid(repeat: int = 3) -> list:
    eta_grid = np.linspace(0, 100, 101, dtype=int)
    eta_grid = np.array(list(product(eta_grid, repeat=repeat)))
    eta_grid = eta_grid[eta_grid.sum(axis=1) == 100]
    eta_grid = [eta for eta in eta_grid]
    return eta_grid


def get_bothsides_eta_grid(repeat: int = 3) -> list:
    eta_grid = np.linspace(-5, 5, 11, dtype=int)
    eta_grid = np.array(list(product(eta_grid, repeat=repeat)))
    eta_grid = eta_grid / np.abs(eta_grid).sum(axis=1)[:, None]
    eta_grid = eta_grid[np.isfinite(eta_grid).all(axis=1)]
    # drop if sum equals 0
    eta_grid = eta_grid[eta_grid.sum(axis=1) != 0]
    eta_grid = np.unique(eta_grid, axis=0)
    return eta_grid


### Functions to load processed results ###
def get_simple_results():
    if os.path.exists("results/simple.pkl"):
        return pd.read_pickle("results/simple.pkl")
    else:
        return pd.Series(
            index=pd.MultiIndex.from_product(
                [
                    get_dataset_names(),  # Dataset names
                    [0, 1, 2],  # LayerID
                    [  # Scores
                        "adamic_adar",
                        "common_neighbors",
                        "has_common_neighbors",
                        "jaccard_coefficient",
                        "preferential_attachment",
                        "multiplex_adamic_adar",
                    ],
                    ["One-Layer", "Simple-Average"],  # Aggregation
                    ["roc_auc"],  # Metric
                ],
                names=["dataset", "layerID", "score", "aggregation", "metric"],
            ),
            dtype=float,
        )


def get_weighted_etas():
    if os.path.exists("results/weighted_etas.pkl"):
        return pd.read_pickle("results/weighted_etas.pkl")
    else:
        return pd.Series(
            index=pd.MultiIndex.from_product(
                [
                    get_dataset_names(),  # Dataset names
                    [0, 1, 2],  # LayerID
                    [  # Scores
                        "adamic_adar",
                        "common_neighbors",
                        "has_common_neighbors",
                        "jaccard_coefficient",
                        "preferential_attachment",
                        "multiplex_adamic_adar",
                    ],
                    ["roc_auc"],  # Metric
                    range(len(get_positive_eta_grid())),
                ],
                names=["dataset", "layerID", "score", "metric", "etaID"],
            ),
            dtype=int,
        )


def get_bothside_etas():
    if os.path.exists("results/bothside_etas.pkl"):
        return pd.read_pickle("results/bothside_etas.pkl")
    else:
        return pd.Series(
            index=pd.MultiIndex.from_product(
                [
                    get_dataset_names(),  # Dataset names
                    [0, 1, 2],  # LayerID
                    [  # Scores
                        "adamic_adar",
                        "common_neighbors",
                        "has_common_neighbors",
                        "jaccard_coefficient",
                        "preferential_attachment",
                        "multiplex_adamic_adar",
                    ],
                    ["roc_auc"],  # Metric
                    range(len(get_bothsides_eta_grid())),
                ],
                names=["dataset", "layerID", "score", "metric", "etaID"],
            ),
            dtype=int,
        )
