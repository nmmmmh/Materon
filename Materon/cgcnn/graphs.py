"""CGCNN graph builders.

* Standard CGCNN: atomic numbers and all periodic neighbours within 8 A (kgcnn input).
* Physics/Materon CGCNN: named attributes (full20, grouped_pca or shuffled_full20),
  at most 12 nearest periodic neighbours within 8 A, 21 edge features
  (distance, 16 Gaussian radial terms, spherical harmonics l <= 1).
  The Materon variant also returns the named node contexts (coordination, q_eff,
  electrostatic magnitude) used only by the context ablations.
"""
from __future__ import annotations

import numpy as np

from materon.attributes import map_atomic_numbers


class GaussRBF:
    def __init__(self, bins=16, cutoff=8.0):
        centers = np.linspace(0.0, cutoff, bins + 1)
        self.mu = centers[1:].astype(np.float32)
        self.sigma = np.diff(centers).astype(np.float32)

    def __call__(self, distance):
        d = np.asarray(distance, np.float32).reshape(-1, 1)
        return np.exp(-((d - self.mu[None, :]) ** 2) / (2 * (self.sigma[None, :] ** 2 + 1e-12))).astype(np.float32)


def spherical_harmonics_l01(offset):
    off = np.asarray(offset, np.float32)
    if off.ndim == 1:
        off = off.reshape(1, 3)
    x, y, z = off[:, 0], off[:, 1], off[:, 2]
    r = np.linalg.norm(off, axis=1) + 1e-12
    c0 = 0.5 * np.sqrt(1 / np.pi)
    c1 = np.sqrt(3 / (4 * np.pi))
    return np.stack([np.full_like(r, c0), c1 * y / r, c1 * z / r, c1 * x / r], axis=-1).astype(np.float32)


def safe_en(z):
    try:
        from pymatgen.core import Element
        return float(Element.from_Z(int(z)).X or 0.0)
    except Exception:
        return 0.0


def structure_neighbors(structure, radius=8.0):
    """Directed periodic neighbours (center, neighbour, distance, image), sorted deterministically."""
    try:
        center, point, image, distance = structure.get_neighbor_list(
            r=float(radius), numerical_tol=1e-8, exclude_self=True)
        records = [(int(i), int(j), float(d), tuple(int(x) for x in img))
                   for i, j, img, d in zip(center, point, image, distance)]
    except Exception:
        records = []
        for i, neighbours in enumerate(structure.get_all_neighbors(
                float(radius), include_index=True, include_image=True)):
            for item in neighbours:
                d = float(item.nn_distance)
                if d > 1e-8:
                    records.append((i, int(item.index), d, tuple(int(x) for x in item.image)))
    records.sort(key=lambda x: (x[0], round(x[2], 12), x[1], x[3]))
    return records


def build_cgcnn_graph(structure, feature_mode="full20", shuffle_seed=None,
                      radius=8.0, max_neighbors=12, rbf_bins=16, include_context=True):
    z = np.asarray(structure.atomic_numbers, dtype=np.int32)
    node_features, bundle = map_atomic_numbers(z, feature_mode, shuffle_seed)
    grouped = {i: [] for i in range(len(z))}
    for record in structure_neighbors(structure, radius):
        grouped[record[0]].append(record)
    edge_records = []
    for i in range(len(z)):
        edge_records.extend(sorted(grouped[i], key=lambda x: (round(x[2], 12), x[1], x[3]))[:int(max_neighbors)])

    result = {"node_features": node_features, "feature_names": bundle["names"]}
    if include_context:
        # Context is accumulated before edge de-duplication; every directed
        # record contributes to both endpoints (original Materon ordering).
        coordination = np.zeros(len(z), np.float32)
        neighbour_en = [[] for _ in range(len(z))]
        for u, v, _, _ in edge_records:
            coordination[u] += 1.0
            coordination[v] += 1.0
            neighbour_en[u].append(safe_en(z[v]))
            neighbour_en[v].append(safe_en(z[u]))

    # Remove only exact duplicate periodic bonds; distinct images stay distinct.
    seen, dedup = set(), []
    for u, v, d, img in edge_records:
        key = (u, v, round(float(d), 8), tuple(int(x) for x in img))
        if key not in seen:
            seen.add(key)
            dedup.append((u, v, d, img))
    edge_records = dedup
    if edge_records:
        edge_index = np.asarray([[u, v] for u, v, _, _ in edge_records], np.int32)
        dist = np.asarray([d for _, _, d, _ in edge_records], np.float32)
        frac = np.asarray(structure.frac_coords, np.float32)
        lattice = np.asarray(structure.lattice.matrix, np.float32)
        displacement = np.asarray([np.matmul(frac[v] + np.asarray(img, np.float32) - frac[u], lattice)
                                   for u, v, _, img in edge_records], np.float32)
        edge_features = np.concatenate([dist[:, None], GaussRBF(rbf_bins, radius)(dist),
                                        spherical_harmonics_l01(displacement)], axis=-1).astype(np.float32)
    else:
        edge_index = np.zeros((0, 2), np.int32)
        edge_features = np.zeros((0, 1 + rbf_bins + 4), np.float32)
    result.update({"edge_indices": edge_index, "edge_features": edge_features})

    if include_context:
        q_eff = np.zeros(len(z), np.float32)
        en = np.asarray([safe_en(value) for value in z], np.float32)
        span = float(np.max(en) - np.min(en) + 1.0e-8)
        for i in range(len(z)):
            if coordination[i] > 0 and neighbour_en[i]:
                q_eff[i] = -(en[i] - float(np.mean(neighbour_en[i]))) / span
        q_eff = np.clip(q_eff, -1, 1).astype(np.float32)
        emag = (np.tanh(np.abs(q_eff) * 0.5)
                + 0.1 * coordination / (float(np.max(coordination)) + 1.0e-8)).astype(np.float32)
        result.update({"coordination": coordination[:, None], "q_eff": q_eff[:, None],
                       "electrostatic_magnitude": emag[:, None]})
    return result


def build_materon_cgcnn_graph(structure, feature_mode="full20", shuffle_seed=None):
    return build_cgcnn_graph(structure, feature_mode, shuffle_seed, 8.0, 12, 16, include_context=True)


def build_physics_cgcnn_graph(structure, feature_mode="full20", shuffle_seed=None):
    return build_cgcnn_graph(structure, feature_mode, shuffle_seed, 8.0, 12, 16, include_context=False)


def build_standard_cgcnn_graph(structure, radius=8.0):
    records = structure_neighbors(structure, radius)
    edges = np.asarray([[i, j] for i, j, _, _ in records], np.int64)
    images = np.asarray([img for _, _, _, img in records], np.float32)
    if edges.size == 0:
        edges = np.zeros((0, 2), np.int64)
        images = np.zeros((0, 3), np.float32)
    return {"node_number": np.asarray(structure.atomic_numbers, np.int64),
            "node_frac_coordinates": np.asarray(structure.frac_coords, np.float32),
            "range_indices": edges, "graph_lattice": np.asarray(structure.lattice.matrix, np.float32),
            "range_image": images}
