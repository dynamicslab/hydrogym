"""Compare DRL-coupled NekRS runs with the uncontrolled reference (uncontrolled/wall.dat, every 10 steps).

Prints per run: the wall dU/dy from aeroForces (solver, $RUN/wall.dat, every 13 steps), the agreement with the
uncontrolled run at common steps, and the HydroGym reward/dU/dy seen by the controller (client_log.json).
Usage: python3 compare_runs.py runs/zero1500 runs/oppo1500
"""

import json
import sys

import numpy as np

ref = np.loadtxt("uncontrolled/wall.dat")
ref_step = ref[:, 0].astype(int)
ref_dudy = dict(zip(ref_step, ref[:, 3]))
DUDY0 = 13.10  # HydroGym baseline (YAML dUdy)

for run in sys.argv[1:]:
    w = np.loadtxt(f"{run}/wall.dat")
    step, dudy = w[:, 0].astype(int), w[:, 3]
    common = [(s, d, ref_dudy[s]) for s, d in zip(step, dudy) if s in ref_dudy]
    c = np.array([(d, r) for _, d, r in common])
    log = json.load(open(f"{run}/client_log.json"))["log"]
    r = np.array([e["reward"] for e in log])
    cd = np.array([e["dUdy"] for e in log])
    n = len(r)
    last = slice(max(0, n - n // 3), n)  # last third of the run
    print(f"== {run}: {len(step)} solver samples to step {step[-1]}, {n} policy steps")
    if len(c):
        rel = (c[:, 0] - c[:, 1]) / c[:, 1]
        print(f"   vs uncontrolled at {len(c)} common steps: max |rel diff| {np.abs(rel).max():.2e}, "
              f"last {rel[-1]:+.2e}")
    tail = step >= step[-1] * 2 // 3
    print(f"   solver dU/dy: first {dudy[0]:.4f}  mean(last third) {dudy[tail].mean():.4f}  "
          f"-> DR vs 13.10 {1 - dudy[tail].mean() / DUDY0:+.3%}")
    print(f"   controller: mean reward(last third) {r[last].mean():+.4f}, "
          f"dU/dy {cd[last].mean():.4f}, u'rms {np.mean([e['obs_u_rms'] for e in log[last]]):.3f}, "
          f"v'rms {np.mean([e['obs_v_rms'] for e in log[last]]):.3f}, act rms {np.mean([e['act_rms'] for e in log[last]]):.3f}")
    ref_tail = ref[(ref_step >= step[-1] * 2 // 3) & (ref_step <= step[-1]), 3].mean()
    print(f"   uncontrolled same window: dU/dy {ref_tail:.4f} -> DR of this run vs uncontrolled "
          f"{1 - dudy[tail].mean() / ref_tail:+.3%}")
