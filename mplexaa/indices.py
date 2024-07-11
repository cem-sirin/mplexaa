# Native Libraries
from itertools import product
from typing import List, Iterable

# Third Party Libraries
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
import networkx as nx

# Local Libraries
from .multiplex_graph import MultiplexGraphDataset

# Alias for adjacency_matrix
adj = nx.adjacency_matrix
# Ignore division by zero and invalid values
np.seterr(divide="ignore", invalid="ignore")


def adamic_adar(a, d, ncombs) -> np.ndarray:
    """Returns the Adamic-Adar index for the given dataset with shape (n_node_combs,)."""
    d_ = d.copy()
    d_[d_ != 0] = np.log(d_[d_ != 0])
    d_[d_ != 0] = 1 / d_[d_ != 0]
    pred = (a[ncombs[0]] * a[ncombs[1]] * d_).sum(axis=1)
    return pred


def common_neighbors(a, d, ncombs) -> np.ndarray:
    """Returns the number of common neighbors between two nodes with shape (n_node_combs,)."""
    return (a[ncombs[0]] * a[ncombs[1]]).sum(axis=1)


def has_common_neighbors(a, d, ncombs) -> np.ndarray:
    """Returns whether two nodes have common neighbors with shape (n_node_combs,)."""
    return (a[ncombs[0]] * a[ncombs[1]]).sum(axis=1) > 0


def jaccard_coefficient(a, d, ncombs) -> np.ndarray:
    """Returns the Jaccard coefficient for the given dataset with shape (n_node_combs,)."""
    pred = (a[ncombs[0]] * a[ncombs[1]]).sum(axis=1) / (a[ncombs[0]] + a[ncombs[1]]).sum(axis=1)
    pred[np.isnan(pred)] = 0
    return pred


def katz_index(a, d, ncombs, beta=0.1) -> np.ndarray:
    """Returns the Katz index for the given dataset with shape (n_node_combs,)."""
    I = np.eye(a.shape[0])
    K = np.linalg.inv(I - beta * a) - I
    return K[ncombs[0], ncombs[1]]


def preferential_attachment(a, d, ncombs) -> np.ndarray:
    """Returns the Preferential Attachment index for the given dataset with shape (n_node_combs,)."""
    return (d[ncombs[0]] - a[ncombs[0], ncombs[1]]) * (d[ncombs[1]] - a[ncombs[0], ncombs[1]])


def multiplex_adamic_adar(
    ds: MultiplexGraphDataset, eta: np.ndarray = None, layers: Iterable[int] = None
) -> List[csr_matrix]:
    """
    Returns the Multiplex Adamic-Adar (MAA) index for the given dataset.

    Args:
        mg: A MultiGraphDataset object
        eta: A matrix of layer weights [L, L]. If None, layer combinations are equally weighted.
        layers: The layers to consider. If None, all layers are considered.

    Returns:
        A [L, n_nodes, n_nodes] array of the MAA index for each layer.
    """
    L = ds.L
    lprod = ds.lprod_

    try:  # Check if the values are already computed
        lprod = ds.lprod_
        degrees = ds.degrees_
    except AttributeError:  # If not, compute them
        mg = ds.mg  # Multiplex Graph
        D = dict()  # Degree matrix for each layer
        k = dict()  # Average degree of each layer

        for l in ds.layer_dict.keys():
            D[l] = np.array(mg[l].degree())[:, 1]
            k[l] = D[l].mean()

        # We iterate over layer permutations and node combinations
        inter_layer_neighbors = dict()
        for alpha, beta in lprod:
            # Inter-layer neighbors shape (n_ncombs, n_nodes)
            inter_layer_neighbors[(alpha, beta)] = adj(mg[alpha])[ds.ncombs_[0]] * adj(mg[beta])[ds.ncombs_[1]]

        degrees = {}  # [lprod, n_node_combs, n_nodes]
        for alpha, beta in lprod:
            # dlog shape (n_nodes,)
            d = np.log(D[alpha]) * np.log(D[beta]) * k[alpha] * k[beta]
            d = d**-0.5
            d[np.isinf(d)] = 0
            d = inter_layer_neighbors[(alpha, beta)] * d
            d = d.sum(axis=1)  # (n_node_combs,)
            d = csr_matrix((d, (ds.ncombs_[0], ds.ncombs_[1])), shape=(len(ds.node_list), len(ds.node_list)))
            degrees[(alpha, beta)] = d
            del d

        # We store the  values
        ds.degrees_ = degrees

    if eta is None:
        eta = np.ones((L, L)) / L

    if layers is None:
        layers = range(L)

    maa = [csr_matrix((len(ds.node_list), len(ds.node_list))) for _ in layers]

    for l, (alpha, beta) in product(layers, lprod):
        maa[layers.index(l)] += eta[l, alpha] * eta[l, beta] * degrees[(alpha, beta)]

    return maa
