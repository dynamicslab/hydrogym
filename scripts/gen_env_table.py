#!/usr/bin/env python3
"""Generate the environment table of HydroGym from the published dataset.

One *environment* is one folder of the Hugging Face dataset
``dynamicslab/HydroGym-environments`` (an environment configuration: flow, Reynolds
number, dimension, mesh/grid variant). The backend of a folder is read from its hidden
sentinel file (``.MAIA_LB``, ``.FIREDRAKE``, ...); folders without a sentinel are
classified by name. A folder is *available* when it holds data beyond its card/config
files, otherwise it is listed as ``coming soon``.

Usage:
    python scripts/gen_env_table.py                       # pinned revision below, prints a summary
    python scripts/gen_env_table.py --revision <sha|tag>  # another revision of the dataset
    python scripts/gen_env_table.py --listing tree.json   # offline: JSON of the Hub API
                                                          # (``siblings`` list) or a list of paths
    python scripts/gen_env_table.py --write               # (re)write docs/docs/environments.md
    python scripts/gen_env_table.py --check               # exit 1 if the generated page or one
                                                          # of the counts in README/docs differs

Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

REPO = "dynamicslab/HydroGym-environments"
# Revision the committed page was generated from. Update together with the release.
PINNED_REVISION = "fbd2800b0b73f0d0182c0c5d71962a51bb5f5902"

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "docs" / "environments.md"

# Backend labels in display order.
BACKENDS = OrderedDict(
    [
        ("firedrake", "Firedrake (FEM)"),
        ("maia_lb", "m-AIA lattice Boltzmann"),
        ("maia_fv", "m-AIA structured finite volume"),
        ("nek5000", "Nek5000"),
        ("jax", "JAX"),
        ("jaxfluids", "JAX-Fluids"),
    ]
)
SENTINELS = {
    ".FIREDRAKE": "firedrake",
    ".MAIA_LB": "maia_lb",
    ".MAIA_STRCTRD": "maia_fv",
    ".NEK5000": "nek5000",
    ".NEK5000_v17": "nek5000",
    ".JAX": "jax",
    ".JAXFLUIDS": "jaxfluids",
}
NOT_ENV = {"policies", "README.md", ".gitattributes"}
META_FILES = {"environment_config.yaml", "environment_config.yml"}
NAME = re.compile(r"^(?P<flow>[A-Za-z0-9]+?)(?P<dim>_[23]D)?_(?P<rest>Re.*)$")


def fetch_listing(revision: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=60) as r:  # public dataset, no token needed
        return [s["rfilename"] for s in json.load(r)["siblings"]]


def load_listing(path: str) -> list[str]:
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data["siblings"]
    return [x["rfilename"] if isinstance(x, dict) else x for x in data]


def classify(folder: str, sentinels: set[str]) -> str:
    for s in sentinels:
        if s in SENTINELS:
            return SENTINELS[s]
    if folder.endswith("_FD"):
        return "firedrake"
    if folder.startswith("Nozzle"):
        return "jaxfluids"
    if folder in ("Kolmogorov_2D", "Channel_3D_Retau180"):
        return "jax"
    return "maia_lb"


def parse(folder: str) -> dict:
    m = NAME.match(folder)
    flow = m.group("flow") if m else folder
    dim = (m.group("dim") or "")[1:] if m else ""
    return {"flow": flow, "dim": dim}


def collect(files: list[str]) -> list[dict]:
    tops: dict[str, list[str]] = defaultdict(list)
    for f in files:
        top, _, rest = f.partition("/")
        if top in NOT_ENV and not rest:
            continue
        if top in NOT_ENV or not rest:
            continue
        tops[top].append(rest)
    envs = []
    for name in sorted(tops):
        rels = tops[name]
        sentinels = {r for r in rels if r.startswith(".") and "/" not in r}
        data = [r for r in rels if r not in sentinels and r not in META_FILES and not r.startswith("README")]
        info = parse(name)
        envs.append(
            {
                "name": name,
                "backend": classify(name, sentinels),
                "files": len(rels),
                "runnable": bool(data),
                **info,
            }
        )
    return envs


def summary(envs: list[dict]) -> dict:
    per = OrderedDict((k, Counter()) for k in BACKENDS)
    for e in envs:
        per[e["backend"]]["total"] += 1
        per[e["backend"]]["runnable" if e["runnable"] else "coming_soon"] += 1
    total = len(envs)
    runnable = sum(1 for e in envs if e["runnable"])
    return {"per": per, "total": total, "runnable": runnable}


def render(envs: list[dict], revision: str) -> str:
    s = summary(envs)
    out = [
        "---",
        "id: environments",
        "title: Environment catalogue",
        "---",
        "",
        "# Environment catalogue",
        "",
        "<!-- Generated by scripts/gen_env_table.py - do not edit by hand. -->",
        "",
        f"This page is generated from the Hugging Face dataset [`{REPO}`](https://huggingface.co/datasets/{REPO}) "
        f"at revision `{revision}`.",
        "One **environment** is one dataset folder: an environment configuration with its own flow, Reynolds number,",
        "dimension and mesh or grid variant. The same flow at two Firedrake mesh resolutions counts as two environments.",
        "An environment is **available** when its folder holds data (grid, restart or checkpoint, case files);",
        "a folder that only holds a placeholder is listed as *coming soon* (its data is not published yet) and is not counted as available.",
        "",
        f"**{s['total']} environment folders, {s['runnable']} available** "
        f"({s['total'] - s['runnable']} coming soon).",
        "",
        "| Solver backend | Environments | Available | Coming soon |",
        "|---|---:|---:|---:|",
    ]
    for k, label in BACKENDS.items():
        c = s["per"][k]
        out.append(f"| {label} | {c['total']} | {c['runnable']} | {c['coming_soon']} |")
    out.append(f"| **All** | **{s['total']}** | **{s['runnable']}** | **{s['total'] - s['runnable']}** |")
    out += [
        "",
        "The NekRS backend has no folders of its own: it runs the Nek5000 turbulent-channel and NACA0012-wing cases",
        "(`TCFlarge_3D_Re207`, `NACA0012_3D_Re200000_AOA0`) and is set up from `examples/nekrs`.",
        "The m-AIA and Nek5000 environments are used through the backend's `from_hf`; the registered `gym.make` ids",
        "are a small subset (see `hydrogym/registration.py`).",
        "",
    ]
    for k, label in BACKENDS.items():
        rows = [e for e in envs if e["backend"] == k]
        if not rows:
            continue
        out += [f"## {label}", "", "| Environment | Flow | Dim | Status |", "|---|---|---|---|"]
        for e in rows:
            out.append(f"| `{e['name']}` | {e['flow']} | {e['dim'] or '-'} | {'available' if e['runnable'] else 'coming soon'} |")
        out.append("")
    return "\n".join(out)


COUNT_FILES = [
    ("README.md", re.compile(r"\*\*(\d+) environments")),
    ("README.md", re.compile(r"HydroGym provides \*\*(\d+) environments")),
    ("docs/docs/introduction.md", re.compile(r"to (\d+) pre-configured CFD environments")),
    ("docs/src/components/HomepageFeatures/index.tsx", re.compile(r"(\d+) pre-configured environments")),
]


def check(envs: list[dict], revision: str) -> int:
    bad = 0
    want = summary(envs)["runnable"]
    page = render(envs, revision)
    if not PAGE.exists() or PAGE.read_text() != page + ("" if page.endswith("\n") else "\n"):
        print(f"MISMATCH {PAGE.relative_to(ROOT)} differs from the generated page")
        bad += 1
    for rel, rx in COUNT_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        for m in rx.finditer(p.read_text()):
            if int(m.group(1)) != want:
                print(f"MISMATCH {rel}: says {m.group(1)}, dataset has {want} available")
                bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--listing", help="offline JSON listing instead of the Hub API")
    ap.add_argument("--write", action="store_true", help=f"write {PAGE.relative_to(ROOT)}")
    ap.add_argument("--check", action="store_true", help="compare page and counts in README/docs")
    ap.add_argument("--list", action="store_true", help="print one line per environment")
    a = ap.parse_args()

    files = load_listing(a.listing) if a.listing else fetch_listing(a.revision)
    envs = collect(files)
    s = summary(envs)
    if a.list:
        for e in envs:
            print(f"{e['name']:45s} {e['backend']:10s} {'available' if e['runnable'] else 'coming soon'}")
    print(f"revision {a.revision}: {s['total']} environment folders, {s['runnable']} available")
    for k, label in BACKENDS.items():
        c = s["per"][k]
        print(f"  {label:34s} {c['total']:3d}  (available {c['runnable']}, coming soon {c['coming_soon']})")
    if a.write:
        text = render(envs, a.revision)
        PAGE.write_text(text if text.endswith("\n") else text + "\n")
        print(f"wrote {PAGE.relative_to(ROOT)}")
    if a.check:
        return 1 if check(envs, a.revision) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
