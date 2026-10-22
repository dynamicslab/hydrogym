"""
Probe and actuator layouts from the YAML description.

A position block is a mapping with a `source`:

  list             points: [[x, y, z], ...]
  file             path: positions.npy | .csv (columns x, y, z) | whitespace text with three columns
  grid             x, y, z: a value, a list, or {start, stop, num}; order: [slowest, ..., fastest] (default x, y, z)
  boundary_nodes   boundary: bid, nodes: all | facindf | unique   (GLL nodes of the boundary, from the solver;
                   'all' keeps the copies of shared edge/corner nodes, as the released HydroGym channel)
  actuator_offset  offset: [dx, dy, dz] and/or normal_distance: d   (probes only: actuator centre + offset)

and an optional `sort: [x, z, ...]` (stable lexicographic sort by those coordinates, first key primary).
"""

from typing import Dict, Optional, Tuple

import numpy as np


def _axis_values(v) -> np.ndarray:
    if isinstance(v, dict):
        return np.linspace(float(v["start"]), float(v["stop"]), int(v["num"]))
    return np.atleast_1d(np.asarray(v, dtype=float))


def _read_file(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).reshape(-1, 3)
    if path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path)
        return df[["x", "y", "z"]].to_numpy(dtype=float)
    return np.loadtxt(path).reshape(-1, 3)


def positions(spec: Dict, iface=None, actuators: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Return (xyz[n, 3], extra columns) for a position block."""
    src = spec["source"]
    extra: Dict[str, np.ndarray] = {}
    if src == "list":
        xyz = np.asarray(spec["points"], dtype=float).reshape(-1, 3)
    elif src == "file":
        xyz = _read_file(spec["path"])
    elif src == "grid":
        order = spec.get("order", ["x", "y", "z"])
        axes = {a: _axis_values(spec[a]) for a in "xyz"}
        mesh = np.meshgrid(*[axes[a] for a in order], indexing="ij")
        cols = {a: m.ravel() for a, m in zip(order, mesh)}
        xyz = np.stack([cols[a] for a in "xyz"], axis=1)
    elif src == "boundary_nodes":
        if iface is None:
            raise ValueError("boundary_nodes needs the solver")
        b = iface.boundary_nodes(int(spec["boundary"]), spec.get("nodes", "all"))
        xyz = b.pop("xyz")
        extra = b
    elif src == "actuator_offset":
        if actuators is None:
            raise ValueError("actuator_offset is only available for probes")
        xyz = actuators["xyz"] + np.asarray(spec.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if "normal_distance" in spec:
            if "normal" not in actuators:
                raise ValueError("normal_distance needs actuators placed on boundary_nodes")
            xyz = xyz + float(spec["normal_distance"]) * actuators["normal"]
        extra = {"actuator": np.arange(len(xyz))}
    else:
        raise ValueError(f"unknown position source '{src}'")

    if "sort" in spec:
        keys = [xyz[:, "xyz".index(k)] for k in reversed(spec["sort"])]
        idx = np.lexsort(keys)  # stable, last key in the list given to lexsort is primary
        xyz = xyz[idx]
        extra = {k: v[idx] for k, v in extra.items()}
    return xyz, extra


def footprint_arrays(spec, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Footprint code and widths (sigma or half width per axis, 0 = unbounded) for n actuators."""
    from .interface import FOOTPRINT_CODES

    spec = spec or {"type": "point"}
    kind = spec["type"]
    if kind == "point":
        w = np.zeros(3)
    elif kind == "gaussian":
        w = np.array([0.0 if s is None else float(s) for s in spec["sigma"]])
    elif kind == "box":
        w = np.array([0.0 if s is None else float(s) for s in spec["half_width"]])
    else:
        raise ValueError(f"unknown footprint '{kind}'")
    return np.full(n, FOOTPRINT_CODES[kind]), np.tile(w, (n, 1))


def direction_arrays(spec, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction code, vector and angle [rad]: 'normal' | {vector: [..]} | {angle: deg, towards: [..]}."""
    from .interface import DIRECTION_CODES

    if spec in (None, "normal"):
        return np.full(n, DIRECTION_CODES["normal"]), np.zeros((n, 3)), np.zeros(n)
    if "vector" in spec:
        return np.full(n, DIRECTION_CODES["vector"]), np.tile(np.asarray(spec["vector"], float), (n, 1)), np.zeros(n)
    if "angle" in spec:
        return (
            np.full(n, DIRECTION_CODES["angle"]),
            np.tile(np.asarray(spec["towards"], float), (n, 1)),
            np.full(n, np.deg2rad(float(spec["angle"]))),
        )
    raise ValueError(f"unknown direction {spec}")
