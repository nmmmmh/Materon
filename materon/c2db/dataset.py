"""C2DB metallic work-function cohort, chemical-system split and prepared-directory access.

Cohort (Sec. S1): source-reported band gap exactly 0 and a finite work function;
ordered, finite, nonsingular structure with a vacuum gap > 10 A; exact duplicate
geometries with the same work function (within 1e-5 eV) keep the first record.
Split: chemical systems (sorted element set) ordered by
sha256('c2db-wf-v1-seed42:' + system); floor(70 %) training, floor(15 %)
validation, remainder test. Labels are not used for the assignment.

A prepared directory contains GRAPHS.pkl.gz, SPLIT.json, TRAIN_/VALIDATION_/
TEST_LABELS.json, EXCLUSIONS.json and PREPARED.json (target scaling, counts).
"""
import gzip
import hashlib
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from materon.c2db.graph import CUTOFF_A, build_graph

EXPECTED_SOURCE_ROWS = 3520
VACUUM_MIN_A = 10.0
DUPLICATE_WF_TOLERANCE_EV = 1e-5
SPLIT_SALT = "c2db-wf-v1-seed42:"
GRAPH_KEYS = ("z", "x", "context", "edges", "distances", "radial", "coords", "images", "lattice")
PARTS = ("train", "validation", "test")


def write_json(path, value, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temp), str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- cohort

def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def read_source(path):
    """JARVIS-format C2DB snapshot (list of records with 'id', 'atoms', 'gap', 'wf')."""
    records = read_json(path)
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_ROWS:
        raise ValueError("expected %d C2DB source rows" % EXPECTED_SOURCE_ROWS)
    ids = [record.get("id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("C2DB source ids must be unique nonempty strings")
    return records


def structure_from_c2db(record):
    atoms = record["atoms"]
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    coords = np.asarray(atoms["coords"], dtype=np.float64)
    elements = atoms["elements"]
    cartesian = atoms["cartesian"]
    if type(cartesian) is not bool:
        raise ValueError("atoms.cartesian must be a boolean")
    if (lattice.shape != (3, 3) or coords.ndim != 2 or coords.shape[1] != 3
            or len(coords) != len(elements) or len(elements) == 0):
        raise ValueError("invalid lattice, coordinates, or site count")
    if not np.all(np.isfinite(lattice)) or not np.all(np.isfinite(coords)):
        raise ValueError("lattice and coordinates must be finite")
    if abs(float(np.linalg.det(lattice))) <= 1e-8:
        raise ValueError("lattice is singular")
    structure = Structure(lattice, elements, coords, coords_are_cartesian=cartesian, to_unit_cell=False)
    if not structure.is_ordered or len(structure) != len(elements):
        raise ValueError("structure is not ordered or site count changed")
    return structure


def vacuum_gap(structure):
    """Largest periodic gap along the cell normal (A)."""
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    normal_vector = np.cross(lattice[0], lattice[1])
    area = float(np.linalg.norm(normal_vector))
    if not math.isfinite(area) or area <= 1e-8:
        raise ValueError("in-plane lattice vectors are singular")
    normal_height = abs(float(np.dot(lattice[2], normal_vector / area)))
    if not math.isfinite(normal_height) or normal_height <= 1e-8:
        raise ValueError("normal cell height is singular")
    fractional_z = np.mod(np.asarray(structure.frac_coords, dtype=np.float64)[:, 2], 1.0)
    fractional_z.sort()
    gaps = np.diff(np.concatenate((fractional_z, [fractional_z[0] + 1.0])))
    return float(np.max(gaps)) * normal_height


def geometry_fingerprint(structure):
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)
    fractional = np.mod(np.asarray(structure.frac_coords, dtype=np.float64), 1.0)
    relative = np.mod(np.round(np.mod(fractional - fractional[0], 1.0), 6), 1.0)
    return (tuple(np.round(lattice, 6).reshape(-1).tolist()),
            tuple(site.specie.symbol for site in structure),
            tuple(tuple(float(value) for value in row) for row in relative))


def reduced_composition(structure):
    counts = Counter(site.specie.symbol for site in structure)
    divisor = math.gcd(*counts.values())
    return "|".join("%s:%d" % (element, counts[element] // divisor) for element in sorted(counts))


def qualify_records(records):
    exclusions, qualified, first_by_fingerprint = [], [], {}
    for source_index, record in enumerate(records, 1):
        record_id = record["id"]
        gap = record.get("gap")
        if not (_finite_number(gap) and float(gap) == 0.0):
            exclusions.append({"source_index": source_index, "source_id": record_id, "reason": "gap_not_exactly_zero"})
            continue
        wf = record.get("wf")
        if not _finite_number(wf):
            exclusions.append({"source_index": source_index, "source_id": record_id,
                               "reason": "work_function_not_finite"})
            continue
        try:
            structure = structure_from_c2db(record)
            vacuum = vacuum_gap(structure)
        except Exception as exc:
            exclusions.append({"source_index": source_index, "source_id": record_id,
                               "reason": "geometry_qualification_failed",
                               "detail": "%s: %s" % (type(exc).__name__, exc)})
            continue
        if not vacuum > VACUUM_MIN_A:
            exclusions.append({"source_index": source_index, "source_id": record_id,
                               "reason": "vacuum_not_greater_than_10_angstrom", "vacuum_gap_angstrom": vacuum})
            continue
        fingerprint = geometry_fingerprint(structure)
        prior = first_by_fingerprint.get(fingerprint)
        if prior is not None:
            difference = abs(float(wf) - prior["wf"])
            if difference > DUPLICATE_WF_TOLERANCE_EV:
                raise ValueError("conflicting WF labels for duplicate geometry: %s and %s" % (prior["id"], record_id))
            exclusions.append({"source_index": source_index, "source_id": record_id,
                               "reason": "duplicate_geometry_same_work_function", "retained_source_id": prior["id"]})
            continue
        elements = sorted({site.specie.symbol for site in structure})
        qualified.append({"id": record_id, "wf": float(wf), "structure": structure,
                          "system": "-".join(elements), "elements": elements,
                          "reduced_composition": reduced_composition(structure)})
        first_by_fingerprint[fingerprint] = {"id": record_id, "wf": float(wf)}
    return qualified, exclusions


def split_by_system(qualified):
    members = defaultdict(list)
    system_by_id, composition_by_id = {}, {}
    for item in qualified:
        system_by_id[item["id"]] = item["system"]
        composition_by_id[item["id"]] = item["reduced_composition"]
        members[item["system"]].append(item["id"])
    ordered = sorted(members, key=lambda s: (hashlib.sha256((SPLIT_SALT + s).encode("utf-8")).hexdigest(), s))
    n_train = int(math.floor(0.70 * len(ordered)))
    n_validation = int(math.floor(0.15 * len(ordered)))
    systems = {"train": ordered[:n_train], "validation": ordered[n_train:n_train + n_validation],
               "test": ordered[n_train + n_validation:]}
    part_of = {system: name for name, values in systems.items() for system in values}
    ids = {name: [] for name in PARTS}
    for record_id, system in system_by_id.items():
        ids[part_of[system]].append(record_id)
    for name in ids:
        ids[name].sort()
    sets = {name: set(composition_by_id[i] for i in ids[name]) for name in PARTS}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise AssertionError("chemical-system split is not disjoint")
    return {
        "ids": ids,
        "system_by_id": dict(sorted(system_by_id.items())),
        "mother_by_id": {record_id: record_id for record_id in sorted(system_by_id)},
        "reduced_composition_by_id": composition_by_id,
        "systems_by_split": {name: sorted(values) for name, values in systems.items()},
        "counts": {"records": {n: len(ids[n]) for n in PARTS}, "systems": {n: len(systems[n]) for n in PARTS}},
    }


def builder_record(item):
    structure = item["structure"]
    sites = [{"species": [{"element": site.specie.symbol, "occu": 1.0}],
              "abc": np.asarray(structure.frac_coords[index], dtype=np.float64).tolist()}
             for index, site in enumerate(structure)]
    return {"id": item["id"], "natoms": len(structure),
            "structure": {"lattice": {"matrix": np.asarray(structure.lattice.matrix).tolist()}, "sites": sites}}


def check_graph(record_id, graph, table):
    """Graph invariants of the cohort: reciprocal edges and no edge across the vacuum."""
    if not np.array_equal(graph["x"], table[graph["z"]]) or graph["context"].shape != (len(graph["z"]), 37):
        raise ValueError("%s: inconsistent node features or context" % record_id)
    keys = {(int(c), int(n), tuple(int(v) for v in im)) for (c, n), im in zip(graph["edges"], graph["images"])}
    if len(keys) != len(graph["edges"]):
        raise ValueError("%s: duplicate periodic edges" % record_id)
    if any((n, c, tuple(-v for v in im)) not in keys for c, n, im in keys):
        raise ValueError("%s: periodic edge lacks reciprocal" % record_id)
    if np.count_nonzero(graph["images"][:, 2] != 0):
        raise ValueError("%s: periodic z-image edges at %.1f A" % (record_id, CUTOFF_A))


def mother_weights(ids, mother_by_id):
    counts = Counter(mother_by_id[uid] for uid in ids)
    w = np.array([1.0 / counts[mother_by_id[uid]] for uid in ids])
    return w / np.mean(w)


def target_scaling(train_ids, labels, mother_by_id):
    """Training mean and population SD (ddof 0); every record has weight 1 here."""
    counts = Counter(mother_by_id[i] for i in train_ids)
    weights = {i: 1.0 / counts[mother_by_id[i]] for i in train_ids}
    factor = len(train_ids) / sum(weights.values())
    weights = {i: value * factor for i, value in weights.items()}
    mean = sum(weights[i] * labels[i] for i in train_ids) / len(train_ids)
    variance = sum(weights[i] * (labels[i] - mean) ** 2 for i in train_ids) / len(train_ids)
    return {"mean": mean, "std": math.sqrt(variance), "std_ddof": 0}


def prepare(source_path, table, out_dir):
    """Write a prepared directory from the C2DB snapshot; never overwrites."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PREPARED.json").exists():
        raise FileExistsError("refusing to overwrite prepared artifacts in %s" % out)
    records = read_source(source_path)
    qualified, exclusions = qualify_records(records)
    split = split_by_system(qualified)
    graphs, labels = {}, {}
    for item in qualified:
        graph = build_graph(builder_record(item), table, cutoff=CUTOFF_A)
        check_graph(item["id"], graph, table)
        graphs[item["id"]] = {name: np.ascontiguousarray(graph[name]) for name in GRAPH_KEYS}
        labels[item["id"]] = item["wf"]
    for part in PARTS:
        write_json(out / ("%s_LABELS.json" % part.upper()), {i: labels[i] for i in split["ids"][part]})
    write_json(out / "SPLIT.json", split)
    write_json(out / "EXCLUSIONS.json", {"counts_by_reason": dict(Counter(e["reason"] for e in exclusions)),
                                         "records": exclusions})
    with (out / "GRAPHS.pkl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            pickle.dump(graphs, compressed, protocol=4)
    scaling = target_scaling(split["ids"]["train"], labels, split["mother_by_id"])
    prepared = {"target": {"name": "wf", "unit": "eV"}, "source_record_count": len(records),
                "record_count": len(graphs), "split_counts": split["counts"], "target_scaling": scaling,
                "cutoff_angstrom": CUTOFF_A}
    write_json(out / "PREPARED.json", prepared)
    return prepared


# ---------------------------------------------------------------- prepared access

class _NumpyCacheReader(pickle.Unpickler):
    """Read graph caches pickled with NumPy 2 in a NumPy 1 runtime."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class Prepared:
    """Read-only access to a prepared directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.split = read_json(self.root / "SPLIT.json")
        scaling = read_json(self.root / "PREPARED.json")["target_scaling"]
        self.mean, self.std = float(scaling["mean"]), float(scaling["std"])
        self._graphs = None

    @property
    def graphs(self):
        if self._graphs is None:
            with gzip.open(self.root / "GRAPHS.pkl.gz", "rb") as stream:
                self._graphs = _NumpyCacheReader(stream).load()
        return self._graphs

    def ids(self, part):
        return self.split["ids"][part]

    def labels(self, part):
        return read_json(self.root / ("%s_LABELS.json" % part.upper()))

    def targets(self, part):
        labels = self.labels(part)
        return np.array([labels[i] for i in self.ids(part)], dtype=np.float64)

    def weights(self, part):
        return mother_weights(self.ids(part), self.split["mother_by_id"])


def metrics(y, prediction, weights=None):
    """Weighted MAE, RMSE and R^2 (weights default to one per record)."""
    weights = np.ones(len(y)) if weights is None else weights
    error = prediction - y
    mean = np.average(y, weights=weights)
    denominator = np.sum(weights * (y - mean) ** 2)
    return dict(mae=float(np.average(np.abs(error), weights=weights)),
                rmse=float(np.sqrt(np.average(error ** 2, weights=weights))),
                r2=float(1 - np.sum(weights * error ** 2) / denominator))


def within_composition_metrics(ids, y, prediction, composition_by_id):
    """Sec. S8: center target and prediction within each composition with >= 2 records; weight 1/n."""
    centered_y, centered_p, weights, groups = [], [], [], 0
    for composition in sorted({composition_by_id[i] for i in ids}):
        index = [j for j, uid in enumerate(ids) if composition_by_id[uid] == composition]
        if len(index) < 2:
            continue
        groups += 1
        centered_y.extend((y[index] - np.mean(y[index])).tolist())
        centered_p.extend((prediction[index] - np.mean(prediction[index])).tolist())
        weights.extend([1.0 / len(index)] * len(index))
    if not groups:
        return None
    return dict(records=len(centered_y), compositions=groups,
                **metrics(np.array(centered_y), np.array(centered_p), np.array(weights)))
