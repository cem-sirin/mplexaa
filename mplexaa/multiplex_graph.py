# Native Libraries
from itertools import product, combinations
from typing import List

# Third Party Libraries
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

# Local Libraries
from .utils import parse_files

# Alias for adjacency_matrix
adj = nx.adjacency_matrix


class MultiplexGraphDataset:
    def __init__(self, dataset: str, layer_filter: List[str] = None, force_undirected: bool = False):
        """
        Initializes a MultiplexGraphDataset object.

        Args:
            dataset (str): The name of the dataset.
            layer_filter (List[str]): A list of layers to filter. If None, all layers are considered.
            force_undirected (bool): If True, the graph is forced to be undirected.
        """
        # Files that contain the layers, nodes and edges paths
        f = parse_files(dataset)
        self.directed = dataset not in ["eua", "csa"]
        if force_undirected:
            self.directed = False

        ### Processing Layers ###
        self.layer_dict = dict(pd.read_csv(f["layers"], sep=" ")["layerLabel"])
        if layer_filter:
            self.layer_dict = {k: v for k, v in self.layer_dict.items() if v in layer_filter}

        ### Processing Edges ###
        edges = pd.read_csv(f["edges"], sep=" ", names=["layer", "u", "v", "weight"]) - 1
        edges.drop("weight", axis=1, inplace=True)  # None of the datasets have weights
        edges = edges[edges["layer"].isin(self.layer_dict.keys())]  # Filter out layers not in the layer_dict
        if not self.directed:
            edges["u"], edges["v"] = edges[["u", "v"]].min(axis=1), edges[["u", "v"]].max(axis=1)
        edges.drop_duplicates(inplace=True)

        # Do self-loops exist?
        self_loops = edges["u"] == edges["v"]
        self.n_self_loops = self_loops.sum()
        edges = edges[~self_loops]

        self.info = []
        for l, df in edges.groupby("layer"):
            degrees = df[["u", "v"]].stack().value_counts()
            self.info.append(
                {
                    "Layer": self.layer_dict[l],
                    "Nodes": df[["u", "v"]].stack().nunique(),
                    "Links": len(df),
                    "⟨k⟩": degrees.mean(),
                    "σ": degrees.std(),
                }
            )

        # Processing Nodes
        self.node_list = sorted(edges[["u", "v"]].stack().unique())
        # Reindex the nodes to be 0-indexed
        edges["u"] = edges["u"].apply(lambda x: self.node_list.index(x))
        edges["v"] = edges["v"].apply(lambda x: self.node_list.index(x))
        self.node_list = sorted(edges[["u", "v"]].stack().unique())

        # Declare graph type
        gtype = nx.DiGraph if self.directed else nx.Graph
        self.mg = {}
        for k, v in self.layer_dict.items():
            self.mg[k] = gtype(name=v)
            self.mg[k].add_nodes_from(self.node_list)

        # Add edges to the graphs
        for _, row in edges.iterrows():
            if row["layer"] in self.mg:
                self.mg[row["layer"]].add_edge(row["u"], row["v"])  # All graphs are unwieghted

        # Reindex the layers to be 0-indexed
        if layer_filter:
            self.layer_dict = dict(enumerate(self.layer_dict.values()))
            self.mg = dict(enumerate(self.mg.values()))

        # L: Number of layers
        # name: Name of the dataset
        # lprod_: All possible combinations of layers
        # ncombs_: All possible combinations of nodes
        self.name = dataset
        self.n_edges = len(edges)
        self.L = len(self.layer_dict)
        self.lprod_ = list(product(self.layer_dict.keys(), repeat=2))
        self.ncombs_ = np.array(list(combinations(self.node_list, 2))).T

    def __repr__(self):
        s = f"{self.name}: {len(self.layer_dict)} layered Multiplex Graph with {len(self.node_list)} nodes and {self.n_edges} {(not self.directed)*'un'}directed edges."
        if self.n_self_loops > 0:
            s += f" Removed {self.n_self_loops} self-loops."
        return s

    def get_info(self) -> pd.DataFrame:
        return pd.DataFrame(self.info)

    def A(self) -> List[csr_matrix]:
        return [adj(v) for v in self.mg.values()]

    def get_degrees(self, layerID: int) -> np.ndarray:
        return np.array(self.mg[layerID].degree())[:, 1]

    def get_true_edges(self, layerID: int):
        true = pd.Series(False, index=[(u, v) for u, v in self.ncombs_.T])
        true[list(self.mg[layerID].edges)] = True
        return true
