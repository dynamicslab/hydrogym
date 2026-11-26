# Released Kolmogorov policies: run and plot

Run the released Kolmogorov policies (`policies/kolmogorov_re40/{ppo,gppo}/seed30`) and the uncontrolled flow in the
HydroGym JAX solver, and plot them in the style of the figures of Lagemann et al., *Nature* 657, 369–376 (2026), [doi:10.1038/s41586-026-10917-6](https://doi.org/10.1038/s41586-026-10917-6).

| file | what it does |
|---|---|
| `run_policy.py` | uncontrolled roll-out and one roll-out per policy from the reset state of the preset; makes the figures |
| `plot_policy.py` | the roll-out figures again from stored npz files (`run_policy.py --store_data`) |

- Environment: `hydrogym.jax.envs.kolmogorov.KolmogorovFlow.extreme_events_re40()` (Re 40, 64 x 64 grid, four
  forcing modes a_i sin(k_i y), k = 4..7, actions in [-0.4, 0.15], 4 time units per action, 100 actions per episode).
- Start: the deterministic reset state of the preset, the same for all roll-outs.
- Policy: deterministic, `mid + half * tanh(mean)` of the actor of `examples/jax/kolmogorov_gppo/ppo.py`, loaded from
  `params.msgpack` with `evaluate.load_params`; the numpy forward pass of `policies/load_policy.py` is checked
  against it on the same observations (`numpy_loader_max_abs_diff` in the printed summary).
- Requirements: the packages of `examples/jax/kolmogorov_gppo/requirements.txt` (JAX, flax, distrax), matplotlib for
  the figures. A CPU is enough: about 2 s per action, i.e. a few minutes per 100-action roll-out. Keep
  `jax_enable_x64` off (float32; do not import `hydrogym.jax.envs.channel` in the same process).
- Platform: JAX uses a GPU when it finds one. The flow is chaotic, so the roll-outs depend on the platform: a run
  on a GPU gives other numbers than a run on a CPU. On a GPU, XLA may also choose other kernels in every new
  process, so two GPU runs of the controlled flow can differ; `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes
  them repeatable. `JAX_PLATFORMS=cpu` runs on the CPU, where repeated runs are bit-identical.

## Quick start

```bash
cd examples/jax/released_policies
python run_policy.py                           # -> kolmogorov_rollout_ppo_vs_gppo.png
python run_policy.py --readable --store_data   # also the readable and per-policy figures, the npz and json files
python plot_policy.py kolmogorov_gppo.npz      # kolmogorov_gppo.png (+ --readable)
python plot_policy.py --compare kolmogorov_ppo.npz kolmogorov_gppo.npz   # the roll-out comparison again
```

Options of `run_policy.py`:

| option | default | meaning |
|---|---|---|
| `--policy` | PPO and GPPO | policy id of `policies/MANIFEST.yaml`; repeat for several |
| `--actions N` | 100 | actions per roll-out (one episode) |
| `--out` | `kolmogorov` | prefix of the output files |
| `--readable` | off | also the readable version and the roll-out figures (incl. actions) of each policy |
| `--store_data` | off | also write the npz and `policy_run.json` of each policy |

## Output

By default one figure (PNG, paper style: panel size, fonts and colours as in the paper, with tick labels, axis
labels and a small legend): `kolmogorov_rollout_ppo_vs_gppo.png`, the roll-outs of the uncontrolled flow, the PPO and
the GPPO policy from the same start state: window-mean dissipation against time.

The roll-outs give the printed summary per policy (mean dissipation, uncontrolled dissipation, TKE, control cost,
reward, mean action). `--readable` adds

- `kolmogorov_rollout_ppo_vs_gppo_readable.png`: a larger version of the figure;
- per policy (`ppo` as example), against the uncontrolled flow from the same start state:
  `kolmogorov_ppo.png` and `kolmogorov_ppo_readable.png` (window-mean dissipation vs time),
  `kolmogorov_ppo_actions_readable.png` (the amplitudes a_1..a_4, shades of the policy colour, a_1 darkest),
  `kolmogorov_ppo_tke_readable.png` (window-mean turbulent kinetic energy).

`--store_data` also writes, per policy,

- `kolmogorov_ppo.npz`: per action and roll-out (`unc_*`, `ctl_*`): `action` (the four amplitudes), `dissipation`
  and `tke` (window means over the 4 saved snapshots of the action), `control_cost` (sum a_i^2), `reward`; `meta`
  (JSON): the policy id, environment, dimension, `n_actions`, `dt_action` and the physical settings (`Re`, action
  bounds);
- `kolmogorov_ppo_policy_run.json`: the policy id, environment, dimension and the number of actions.

`plot_policy.py kolmogorov_ppo.npz` makes the roll-out figures again from a stored file: `kolmogorov_ppo.png`;
`--readable` adds `kolmogorov_ppo_readable.png`, `kolmogorov_ppo_actions_readable.png` and `kolmogorov_ppo_tke_readable.png`
(`--out PREFIX` sets the prefix instead of the npz name). `plot_policy.py --compare kolmogorov_ppo.npz
kolmogorov_gppo.npz` draws `kolmogorov_rollout_ppo_vs_gppo.png` again (`--out` prefix default `kolmogorov`).
Without npz files, `plot_policy.py` prints its usage and exits (the figures need the stored roll-outs).

The figure settings come from the plotting code (`examples/maia/released_policies/policy_cases.py` and
`plot_policy.py`, from the policy id), not from the npz. The y-ranges are fitted to the data.

Colours: uncontrolled `#2E2E2E`, PPO `#1f77b4`, GPPO `#2ca02c`.
