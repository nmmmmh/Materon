"""The 37-entry neighborhood descriptor c_i of the C2DB environment gate (Sec. S2.4).

For neighbors within the cutoff (5 A), w(d) = 0.5 [1 + cos(pi d / 5)]. Entries
0-19 are w-weighted means of the neighbors' standardized attributes, 20-35 are
w-weighted means of 16 Gaussian radial terms (centers 0-5 A, width 5/15 A), and
entry 36 is log(1 + sum w). The descriptor is rotation invariant.
"""
import numpy as np


def radial_context(x, edge_indices, distances, cutoff=5.0, bins=16, eps=1e-8):
    """Return one 37-value environment summary per 20-value node feature row.

    Edges are directed center-neighbor pairs; periodic images should be supplied
    as separate rows, including when their atom indices are repeated.
    """
    x = np.asarray(x, dtype=np.float64)
    edges = np.asarray(edge_indices)
    distances = np.asarray(distances, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 20:
        raise ValueError("x must have shape (n_nodes, 20)")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_indices must be an integer array with shape (n_edges, 2)")
    if distances.ndim != 1 or distances.shape[0] != edges.shape[0]:
        raise ValueError("distances must have one value per edge")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("distances must be finite and nonnegative")
    if edges.size and (np.any(edges < 0) or np.any(edges >= x.shape[0])):
        raise ValueError("edge indices are out of bounds for x")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer of at least 2")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    cutoff = float(cutoff)
    eps = float(eps)
    centers = edges[:, 0]
    neighbors = edges[:, 1]
    weights = np.zeros(len(distances), dtype=np.float64)
    active = distances < cutoff
    weights[active] = 0.5 * (1.0 + np.cos(np.pi * distances[active] / cutoff))

    n_nodes = x.shape[0]
    weight_sum = np.bincount(centers, weights=weights, minlength=n_nodes)
    neighbor_sum = np.zeros((n_nodes, 20), dtype=np.float64)
    np.add.at(neighbor_sum, centers, weights[:, None] * x[neighbors])
    neighbor_mean = neighbor_sum / (weight_sum[:, None] + eps)

    radial_centers = np.linspace(0.0, cutoff, bins, dtype=np.float64)
    sigma = cutoff / (bins - 1)
    rbf = np.exp(-0.5 * ((distances[:, None] - radial_centers[None, :]) / sigma) ** 2)
    radial_sum = np.zeros((n_nodes, bins), dtype=np.float64)
    np.add.at(radial_sum, centers, weights[:, None] * rbf)
    radial_mean = radial_sum / (weight_sum[:, None] + eps)
    count = np.log1p(weight_sum)[:, None]
    return np.concatenate((neighbor_mean, radial_mean, count), axis=1).astype(np.float32)
