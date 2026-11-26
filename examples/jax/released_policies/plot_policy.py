#!/usr/bin/env python3
"""Figures of stored roll-outs of the released Kolmogorov policies in the paper style.

    python plot_policy.py kolmogorov_ppo.npz [--out figs/ppo] [--readable]   # a stored roll-out
    python plot_policy.py --compare kolmogorov_ppo.npz kolmogorov_gppo.npz   # -> kolmogorov_rollout_ppo_vs_gppo.png

The npz files are written by ``run_policy.py --store_data``; without npz files the script prints this usage and
exits. With one or more npz files of one policy: the roll-out figures of the plotting script of the m-AIA examples
(``examples/maia/released_policies/plot_policy.py``; figure settings from the policy id): ``kolmogorov_ppo.png``
(dissipation); ``--readable`` adds ``kolmogorov_ppo_readable.png``, ``kolmogorov_ppo_actions_readable.png`` and
``kolmogorov_ppo_tke_readable.png``. ``--compare`` with the npz files of several policies: their roll-outs and the
uncontrolled one in one figure (as ``run_policy.py``), ``<out>_rollout_ppo_vs_gppo.png`` (``--out`` default
``kolmogorov``; ``--readable`` adds the readable version).
"""

import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "maia", "released_policies")

if __name__ == "__main__":
    if "--compare" in sys.argv[1:]:
        ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        ap.add_argument("--compare", nargs="+", required=True, help="npz files of run_policy.py --store_data")
        ap.add_argument("--out", default="kolmogorov", help="prefix of the output file")
        ap.add_argument("--readable", action="store_true", help="also the larger readable figure")
        a = ap.parse_args()
        sys.path.insert(0, MAIA)
        import plot_policy as P  # the m-AIA examples' plotting module

        runs = []
        for f in a.compare:
            R, meta = P.load_runs([f])
            runs.append((meta, R))
        name = "_vs_".join(m["policy"].split("/")[1] for m, _ in runs)
        for f in P.plot_comparison(runs, f"{a.out}_rollout_{name}", readable=a.readable):
            print(f"wrote {f}")
    elif any(x.endswith(".npz") for x in sys.argv[1:]):
        sys.argv[0] = os.path.join(MAIA, "plot_policy.py")
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:  # no run files: usage
        print(__doc__.strip(), file=sys.stdout if {"-h", "--help"} & set(sys.argv[1:]) else sys.stderr)
        sys.exit(0 if {"-h", "--help"} & set(sys.argv[1:]) else 2)
