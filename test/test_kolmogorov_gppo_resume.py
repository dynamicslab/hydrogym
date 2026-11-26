"""Kolmogorov PPO / GPPO trainers (examples/jax/kolmogorov_gppo): a resumed run equals an uninterrupted one.

Uses a cheap variant of the extreme-event preset (16 x 16 grid, dt 0.01) and a tiny batch, so it runs on a CPU.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("gymnax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("distrax")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "jax" / "kolmogorov_gppo"
sys.path.insert(0, str(EXAMPLES))

import gppo  # noqa: E402
import ppo  # noqa: E402

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow  # noqa: E402


def _cheap_env():
    return KolmogorovFlow.extreme_events_re40(env_config={"params": {"dt": 0.01}}, flow_config={"grid_size": (16, 16)})


def _run(module, tmp, name, **kw):
    cfg = dataclasses.replace(
        module.CONFIG, num_envs=2, num_steps=2, minibatches=2, epochs=1, num_updates=3, out_dir=str(tmp / name), **kw
    )
    return module.train(cfg, env=_cheap_env()), cfg


def _check_resume(module, prefix, tmp, **kw):
    straight, _ = _run(module, tmp, "straight", **kw)
    assert (tmp / "straight" / f"{prefix}_DONE").exists()
    # max_minutes ~ 0 stops after the first update with a checkpoint
    stopped, _ = _run(module, tmp, "resumed", max_minutes=1e-9, **kw)
    assert stopped is None and not (tmp / "resumed" / f"{prefix}_DONE").exists()
    resumed, _ = _run(module, tmp, "resumed", resume=True, **kw)
    assert (tmp / "resumed" / f"{prefix}_DONE").exists()
    for a, b in zip(jax.tree_util.tree_leaves(straight), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    m1 = np.load(tmp / "straight" / f"{prefix}_metrics.npz")
    m2 = np.load(tmp / "resumed" / f"{prefix}_metrics.npz")
    assert list(m1["update"]) == [1, 2, 3] == list(m2["update"])
    for k in ("mean_reward", "mean_dissipation", "mean_control_cost", "loss"):
        np.testing.assert_array_equal(m1[k], m2[k])


def test_ppo_resume_is_exact(tmp_path):
    _check_resume(ppo, "ppo", tmp_path)


def test_ppo_rollout_normalisation_resume(tmp_path):
    _check_resume(ppo, "ppo", tmp_path, adv_norm="rollout")


def test_gppo_resume_is_exact(tmp_path):
    _check_resume(gppo, "gppo", tmp_path)


def test_gppo_beta0_differs_only_by_physics_term(tmp_path):
    """beta 0 runs the same code path; its first-update batch (sampled before any update) equals beta 2's."""
    _run(gppo, tmp_path, "b2")
    _run(gppo, tmp_path, "b0", beta=0.0)
    m2 = np.load(tmp_path / "b2" / "gppo_metrics.npz")
    m0 = np.load(tmp_path / "b0" / "gppo_metrics.npz")
    assert m0["mean_reward"][0] == m2["mean_reward"][0]
    assert not np.array_equal(m0["loss"], m2["loss"])
