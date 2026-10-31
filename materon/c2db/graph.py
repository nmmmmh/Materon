"""Periodic graphs, model inputs and fixed-geometry substitutions for the C2DB study.

A graph is a dict of NumPy arrays:
z (atomic numbers), x (20 standardized attributes), context (37 entries),
edges (directed center-neighbor pairs, uncapped within 5 A), distances, radial
(16 Gaussian terms), coords (Cartesian), images (periodic image of each edge)
and lattice.
"""
import numpy as np
from pymatgen.core import Structure

from materon.c2db.context import radial_context

ARM_NAMES = (
    "atomic_number_embedding",
    "physical20_nogate",
    "physical20_global_gate",
    "physical20_element_gate",
    "physical20_environment_gate",
)
GATE_ARMS = ("physical20_global_gate", "physical20_element_gate", "physical20_environment_gate")
DEFAULT_ARM = "physical20_environment_gate"
CUTOFF_A = 5.0


def structure_from_record(record):
    """Ordered pymatgen Structure from {'natoms', 'structure': {'lattice': {'matrix'}, 'sites'}}."""
    natoms = record["natoms"]
    if not isinstance(natoms, (int, np.integer)) or isinstance(natoms, (bool, np.bool_)) or natoms < 1:
        raise ValueError("natoms must be a positive integer")
    data = record["structure"]
    lattice_data = data["lattice"]
    if "pbc" in lattice_data and tuple(lattice_data["pbc"]) != (True, True, True):
        raise ValueError("lattice pbc must be [True, True, True]")
    lattice = np.asarray(lattice_data["matrix"], dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice matrix must be finite with shape (3, 3)")
    if abs(np.linalg.det(lattice)) <= 1e-8:
        raise ValueError("lattice matrix is singular")
    sites = data["sites"]
    if len(sites) != natoms:
        raise ValueError("site count does not match natoms")
    species, coords = [], []
    for site in sites:
        components = site["species"]
        if len(components) != 1 or float(components[0]["occu"]) != 1.0:
            raise ValueError("each site must have one fully occupied species")
        species.append(components[0]["element"])
        abc = np.asarray(site["abc"], dtype=np.float64)
        if abc.shape != (3,) or not np.all(np.isfinite(abc)):
            raise ValueError("fractional coordinates must be finite three-vectors")
        coords.append(abc)
    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    if len(structure) != natoms or not structure.is_ordered:
        raise ValueError("structure must contain natoms ordered sites")
    return structure


def build_graph(record, table, cutoff=CUTOFF_A):
    """Uncapped directed periodic graph within ``cutoff``; zero-distance pairs are omitted."""
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    structure = structure_from_record(record)
    table = np.asarray(table, dtype=np.float32)
    if table.shape != (101, 20) or not np.all(np.isfinite(table)):
        raise ValueError("table must be a finite (101, 20) atomic feature array")
    z = np.asarray(structure.atomic_numbers, dtype=np.int64)
    if np.any(z < 1) or np.any(z > 100):
        raise ValueError("structure contains an element without a feature row")
    x = table[z].copy()

    centers, neighbors, images, distances = structure.get_neighbor_list(cutoff)
    rows, seen = [], set()
    for center, neighbor, image, distance in zip(centers, neighbors, images, distances):
        center, neighbor = int(center), int(neighbor)
        image = tuple(int(value) for value in image)
        key = (center, neighbor, image)
        if key in seen:
            raise ValueError("duplicate periodic edge %r" % (key,))
        seen.add(key)
        distance = float(distance)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("neighbor distances must be finite and nonnegative")
        if distance == 0.0:
            continue
        rows.append((center, distance, neighbor, image))
    rows.sort(key=lambda edge: (edge[0], edge[1], edge[2], edge[3]))

    edges = np.asarray([(edge[0], edge[2]) for edge in rows], dtype=np.int64).reshape((-1, 2))
    distances = np.asarray([edge[1] for edge in rows], dtype=np.float32)
    images = np.asarray([edge[3] for edge in rows], dtype=np.int64).reshape((-1, 3))
    neighbor_counts = np.bincount(edges[:, 0], minlength=len(structure))
    neighborless = np.flatnonzero(neighbor_counts == 0)
    if neighborless.size:
        raise ValueError("neighborless node(s) at cutoff %g: %s" % (cutoff, neighborless.tolist()))

    coords = np.asarray(structure.cart_coords, dtype=np.float32)
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)
    displacement = coords[edges[:, 1]] + images @ lattice - coords[edges[:, 0]]
    if not np.allclose(np.linalg.norm(displacement, axis=1), distances, atol=2e-5, rtol=2e-5):
        raise ValueError("neighbor distances do not match periodic displacement geometry")

    context = radial_context(x, edges, distances, cutoff=cutoff, bins=16)
    centers = np.linspace(0.0, cutoff, 16, dtype=np.float64)
    sigma = cutoff / 15.0
    radial = np.exp(-0.5 * ((distances[:, None] - centers[None, :]) / sigma) ** 2).astype(np.float32)
    graph = dict(z=z, x=x, context=context, edges=edges, distances=distances, radial=radial,
                 coords=coords, images=images, lattice=lattice)
    if any(not np.all(np.isfinite(value)) for value in graph.values()):
        raise ValueError("graph contains nonfinite arrays")
    return graph


def copy_with_features(graph, table, replacement_index=None, replacement_z=None):
    """Copy of ``graph`` with site ``replacement_index`` set to element ``replacement_z``.

    Geometry, edges and radial terms are unchanged (parent geometry); x and the
    context are recomputed from the attribute table, so the chemical context of
    every neighbor of the substituted site changes while its own x does not.
    """
    result = {key: np.asarray(value).copy() for key, value in graph.items()}
    result["z"] = np.asarray(graph["z"], dtype=np.int64).copy()
    if replacement_index is not None:
        result["z"][replacement_index] = replacement_z
    result["x"] = table[result["z"]].copy()
    result["context"] = radial_context(result["x"], result["edges"], result["distances"], cutoff=5.0, bins=16)
    return result


def model_inputs(graphs, arm=DEFAULT_ARM, mask=None):
    """Batched ragged MEGNet inputs for one model arm.

    ``mask`` (gate arms only) is None (ones), a scalar in [0, 1], or one array per
    graph shaped like graph['x']; it scales the gate correction per atom and attribute.
    """
    import tensorflow as tf

    if not graphs:
        raise ValueError("graphs must contain at least one structure")
    if arm not in ARM_NAMES:
        raise ValueError("arm must be one of %s" % (", ".join(ARM_NAMES),))
    if mask is not None and arm not in GATE_ARMS:
        raise ValueError("mask overrides apply only to gate arms")

    def ragged(arrays, dtype=tf.float32):
        values = tf.convert_to_tensor(np.concatenate(arrays, axis=0), dtype=dtype)
        lengths = np.asarray([len(array) for array in arrays], dtype=np.int64)
        return tf.RaggedTensor.from_row_lengths(values, lengths)

    if arm == "atomic_number_embedding":
        atomic_numbers = []
        for graph in graphs:
            z = np.asarray(graph["z"])
            if (z.ndim != 1 or len(z) != len(graph["x"]) or not np.issubdtype(z.dtype, np.integer)
                    or np.any(z < 0) or np.any(z > 100)):
                raise ValueError("graph['z'] must be an integer vector aligned to nodes in [0, 100]")
            atomic_numbers.append(z.astype(np.int32, copy=False))
        args = [ragged(atomic_numbers, tf.int32)]
    else:
        node = ragged([graph["x"] for graph in graphs])
        if arm == "physical20_nogate":
            args = [node]
        else:
            if mask is None:
                masks = [np.ones_like(graph["x"], dtype=np.float32) for graph in graphs]
            elif np.isscalar(mask):
                value = float(mask)
                if not np.isfinite(value) or value < 0.0 or value > 1.0:
                    raise ValueError("scalar mask must be finite and in [0, 1]")
                masks = [np.full_like(graph["x"], value, dtype=np.float32) for graph in graphs]
            else:
                if len(graphs) == 1 and np.asarray(mask).shape == graphs[0]["x"].shape:
                    supplied = [mask]
                else:
                    supplied = list(mask)
                if len(supplied) != len(graphs):
                    raise ValueError("mask must contain one array per graph")
                masks = []
                for graph, values in zip(graphs, supplied):
                    values = np.asarray(values, dtype=np.float32)
                    if values.shape != graph["x"].shape:
                        raise ValueError("each mask must match its graph x shape")
                    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                        raise ValueError("mask values must be finite and in [0, 1]")
                    masks.append(values)
            mask_tensor = ragged(masks)
            if arm in ("physical20_global_gate", "physical20_element_gate"):
                args = [node, mask_tensor]
            else:
                args = [node, ragged([graph["context"] for graph in graphs]), mask_tensor]
    edges = ragged([graph["edges"] for graph in graphs], tf.int64)
    return args + [ragged([graph["coords"] for graph in graphs]), edges,
                   tf.zeros((len(graphs), 1)),
                   ragged([graph["images"] for graph in graphs], tf.int64),
                   tf.constant(np.stack([graph["lattice"] for graph in graphs]), dtype=tf.float32)]
