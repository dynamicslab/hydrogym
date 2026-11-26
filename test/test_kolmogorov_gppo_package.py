"""The Kolmogorov PPO/GPPO example (examples/jax/kolmogorov_gppo) runs end to end on a CPU with tiny settings.

- the four variants train (cheap 16 x 16 env, 2 updates) into the run-dir layout of run_all.sh;
- evaluate.py --quick runs every task, including the policies just trained;
- analyse.py produces results.json with the measures of the README from synthetic 100-update logs.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "jax" / "kolmogorov_gppo"
ROOT = EX.parents[2]
VARIANTS = {
    "ppo_mb": ("ppo", {}),
    "ppo_ro": ("ppo", {"adv_norm": "rollout"}),
    "gppo_b0": ("gppo", {"beta": 0.0}),
    "gppo_b2": ("gppo", {"beta": 2.0}),
}


def _jax_modules():
    pytest.importorskip("jax")
    pytest.importorskip("gymnax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    sys.path.insert(0, str(EX))
    import gppo
    import ppo

    return {"ppo": ppo, "gppo": gppo}


def _cheap_env():
    from hydrogym.jax.envs.kolmogorov import KolmogorovFlow

    return KolmogorovFlow.extreme_events_re40(env_config={"params": {"dt": 0.01}}, flow_config={"grid_size": (16, 16)})


def test_train_evaluate_end_to_end(tmp_path):
    mods = _jax_modules()
    for v, (algo, kw) in VARIANTS.items():
        cfg = dataclasses.replace(
            mods[algo].CONFIG,
            seed=30,
            num_envs=2,
            num_steps=2,
            minibatches=2,
            epochs=1,
            num_updates=2,
            out_dir=str(tmp_path / "runs" / v / "s30"),
            **kw,
        )
        mods[algo].train(cfg, env=_cheap_env())
        assert (tmp_path / "runs" / v / "s30" / f"{algo}_DONE").exists()
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT)}
    cmd = [
        sys.executable,
        str(EX / "evaluate.py"),
        "--quick",
        "--out",
        str(tmp_path / "eval"),
        "--policy-root",
        str(ROOT / "policies" / "kolmogorov_re40"),
        "--with-trained",
        "--runs-root",
        str(tmp_path / "runs"),
        "--minutes",
        "60",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=tmp_path)
    st = subprocess.run(cmd + ["--status"], check=True, env=env, cwd=tmp_path, capture_output=True, text=True)
    assert st.stdout.strip().splitlines()[-1] == "complete"
    import pickle

    rt = pickle.load(open(tmp_path / "eval" / "trained_gppo_b2.pkl", "rb"))
    assert rt["recs"]["dissipation"].shape[0] == 5  # 4 bank states + the reset state, quick sizes


def test_analyse_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(EX))
    import analyse

    monkeypatch.setattr(analyse, "NB", 200)
    monkeypatch.setattr(analyse, "NB_S1", 50)
    rng = np.random.default_rng(0)
    u = np.arange(1, 101)
    for v, slope in (("ppo_mb", 0.0005), ("ppo_ro", 0.0008), ("gppo_b0", 0.0008), ("gppo_b2", 0.0015)):
        algo = VARIANTS[v][0]
        for s in (30, 31, 32):
            d = tmp_path / "runs" / v / f"s{s}"
            d.mkdir(parents=True)
            r = -0.92 + slope * u + 0.01 * rng.standard_normal(100)
            np.savez(
                d / f"{algo}_metrics.npz",
                update=u,
                env_steps=40 * u,
                mean_reward=r,
                mean_dissipation=0.146 - 0.3 * slope * u,
                mean_control_cost=0.09 + 0 * u,
                loss=0 * r,
                wall_s=40 + 0 * r,
            )
    res = analyse.main(tmp_path, tmp_path / "analysis")
    out = json.loads((tmp_path / "analysis" / "results.json").read_text())
    assert out["S1_reward"]["U_gppo_b2"] < out["S1_reward"]["U_ppo_mb"] <= 100  # R* is reached by construction
    assert out["effects_final"]["mean_reward"]["total_G2_minus_P0"]["estimate"] > 0
    assert (tmp_path / "analysis" / "learning_curves.png").exists()
    assert res["seeds"]["gppo_b2"] == [30, 31, 32]
