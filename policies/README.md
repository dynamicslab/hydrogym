# Policies

This folder holds policies for HydroGym environments:

- `<case>/<variant>/seed<k>/`: the policy files;
- `MANIFEST.yaml`: the index of all policies, read by the loader;
- `load_policy.py`: a numpy-only loader (numpy and PyYAML, no solver code);
- `probes/`: observation probe coordinates.

## Index

`MANIFEST.yaml` has one entry per policy id with:

- `id` (`<case>/<variant>/seed<k>`) and `group`;
- `hydrogym_env`: the HydroGym environment id (`id_3d`: the 3D environment the same policy also runs in), the
  backend, and `config`, the environment settings the policy runs with on top of the environment's default
  configuration (e.g. `max_control`, `num_sim_substeps_per_actuation`);
- `observation.probes_file`: the probe coordinates of the observation, where the policy has its own probe file;
- `action.scale_2d_to_3d` (3D pinball): the 3D command is `clip(scale_2d_to_3d * a, -1, 1)`;
- `loader`: the file format, the role of the file the loader reads and, where set, `max_control`
  (command = `a * max_control`);
- `files`: role, path, sha256, size and whether the file is tracked in git.

Variant names: `ppo`, `td3`, `gppo` name the training algorithm; `_tuned` marks PPO policies with
case-specific training settings; `td3_long` is a TD3 policy from a longer training run. On the NACA0012 wing, `td3`
is the TD3 policy of the HydroGym paper (`eval_model_lc_omega1_SS_BASE_dr.zip` keeps the file name of its training
framework), `td3_N1` a TD3 wall policy trained with the actuation power in the reward, and `td3_N1F` the same with
a second actor (`F`) for the upstream part of the actuated strip.

## Loading

```python
from policies.load_policy import load_policy

pi = load_policy("pinball_2d_re150/ppo/seed2")
a = pi(obs)  # obs (83,) = [u/U (40), v/U (40), previous action (3)] -> a in [-1, 1]^3
omega = a * pi.max_control  # rotation rates
```

```bash
python policies/load_policy.py --list                       # id, environment, format of every entry
python policies/load_policy.py --check                      # files present, sha256, one forward pass per entry
python policies/load_policy.py pinball_2d_re150/ppo/seed2   # interface of an entry
```

The docstring at the top of `load_policy.py` defines the forward pass of each format. Stateful formats need
`pi.reset()` at the start of an episode: the cavity formats (rate limit and frame stack) and the pinball format
`pinball_residual_npz` (action counter).

The pickles (`*.pkl`) are plain tuples `(params, norm)` of numpy arrays that unpickle without JAX. The `npz` files
(gust, cavity, Kolmogorov, wing, pinball `ppo_power`) hold the same weights in a framework-neutral form.

## Running

`examples/maia/released_policies/` runs a policy by its id in its m-AIA environment against the uncontrolled flow
and plots both roll-outs; `examples/jax/released_policies/` does the same for the Kolmogorov policies:

```bash
cd examples/maia/released_policies
./run_example.sh pinball_2d_re100/ppo_tuned/seed0 runs/pinball_re100 1
```

`examples/nekrs/wing` runs the NACA0012 wing policies in NekRS (`./run.sh --policy td3:naca0012_wing_re2e5/td3/seed1998`,
see its README).

## Policy ids

| case | environment | variants (seed) | format |
|---|---|---|---|
| `pinball_2d_re100` | `Pinball_2D_Re100` | `ppo_tuned` (0), `ppo_IM` (0) | `jax_ppo_pkl` |
| `pinball_2d_re100` | `Pinball_2D_Re100` | `ppo_power` (0) | `pinball_residual_npz` |
| `pinball_2d_re150` | `Pinball_2D_Re150` | `ppo` (2) / `td3_long` (2) | `jax_ppo_pkl` / `jax_td3_pkl` |
| `pinball_2d_re150` | `Pinball_2D_Re150` | `ppo_power` (1) | `pinball_residual_npz` |
| `pinball_3d_re150` | `Pinball_3D_Re150` | `ppo_IM` (0), `ppo_tuned` (0) | `jax_ppo_pkl` |
| `cylinder_2d_re200` | `Cylinder_2D_Re200` | `ppo` (1) | `jax_ppo_pkl` |
| `cylinder_2d_re1000` | `Cylinder_2D_Re1000` | `ppo` (2), `ppo_finetune_from_re200` (2) | `jax_ppo_pkl` |
| `cylinder_2d_re3900` | `Cylinder_2D_Re3900` | `ppo` (1), `ppo_finetune_from_re1000` (2) | `jax_ppo_pkl` |
| `square_cylinder_2d_re200` | `SquareCylinder_2D_Re200` | `ppo` (0), `ppo_finetune_from_cylinder_re200` (2) | `jax_ppo_pkl` |
| `naca0012_gust_re1000` | `NACA0012Gust_{2D,3D}_Re1000_AOA20` | `td3_3jet_jet1.5U` (1), `td3_2jet_jet1.5U` (0) | `gust_deploy_npz` |
| `cavity_2d_re4200` | `Cavity_2D_Re4200` | `ppo_IM` (0), `ppo_tuned` (0) | `cavity_spec_npz` |
| `cavity_2d_re7500` | `Cavity_2D_Re7500`, `Cavity3Jet_2D_Re7500` | `ppo_1jet` (0), `ppo_3jet` (1) | `cavity_spec_npz` |
| `cavity_3d_re7500` | `Cavity_3D_Re7500_lz024`, `Cavity3Jet_3D_Re7500_lz024` | `ppo_IM_1jet` (1), `ppo_IM_3jet` (3) | `cavity3d_npz` |
| `cavity_3d_re4200` | `Cavity_3D_Re4200_lz024`, `Cavity3Jet_3D_Re4200_lz024` | `ppo_IM_1jet` (11), `ppo_IM_3jet` (13) | `cavity3d_npz` |
| `kolmogorov_re40` | `KolmogorovFlow.extreme_events_re40()` (JAX) | `ppo` (30), `gppo` (30) | `kolmogorov_npz` |
| `naca0012_wing_re2e5` | `NACA0012_3D_Re200000_AOA0` (NekRS, `examples/nekrs/wing`) | `td3` (1998), `td3_N1` (0), `td3_N1F` (3) | `wall_mlp_npz` / `wall_blocks` |

`python policies/load_policy.py --list` prints every id.

## Settings

- **Pinball 2D Re 100** (`ppo_tuned`, `ppo_IM`): `max_control` 0.3125; all other settings as in `Pinball_2D_Re100`.
- **Pinball 2D Re 150:** `max_control` 0.4, `num_sim_substeps_per_actuation` 152.
- **Pinball 2D `ppo_power`:** Re 100 `max_control` 0.25, Re 150 as above. `pinball_residual_npz`: obs (92,) =
  [u/U (40), v/U (40), previous action (3), C_D, C_L, C_P of the three cylinders (9)]; `hold_actions` > 0: the base
  rotation is held for that many actions after `reset()` (`examples/maia/released_policies/README.md`).
- **Pinball 3D:** the policy output `a` is in 2D units; the 3D command is `a3 = clip(1.6 a, -1, 1)` and the
  rotation rates are `a3 * max_control`. The previous-action input is `a3 / 1.6`.
  The policies run in the default configuration of `Pinball_3D_Re150` (Ma 0.1, `max_control` 0.35,
  `num_sim_substeps_per_actuation` 352), from the environment restart.
- **Cylinder:** `max_control` 0.0173 (0.3 U); 88 (Re 200), 224 (Re 1000) and 496 (Re 3900) lattice steps per action.
- **Square cylinder:** `max_control` 0.021 (0.36 U), the bound of `SquareCylinder_2D_Re200`.
- **NACA0012 wing:** `examples/nekrs/wing/wing.yml`. One actor per wall actuator: obs [u_t', v_n'] / u_tau(x) at
  y+ = 15 -> a in [-1, 1], the wall-normal velocity in units of u_tau,ch = 0.0624334733. `td3` is the
  Stable-Baselines3 policy of the HydroGym paper (the zip, and `actor.npz` for the numpy loader); `td3_N1F` applies
  its F actor on 0.2 <= x/c < 0.48 and N1 on 0.48 <= x/c <= 0.9 (`pi(obs, x)`).
- **Gust airfoil:** the controller file of each target case (`deploy_2d`, `deploy_3d_at*`) holds `max_control`,
  the lattice steps per action and the probe coordinates. `NACA0012Gust_{2D,3D}_Re1000_AOA20` default to
  `max_control` = 1.5 U (0.0866 / 0.1732 lattice), the bound of the `*_jet1.5U` policies. Gust field: `fig2c` (the `transverse` gust of `NACA0012Gust`).

## Probes

- `pinball_2d_wake_probes.json`: the 40 pinball wake probes, in cylinder diameters.
- `pinball_3d_wake_probes.json`: the 160 probes of the 3D pinball policies: the 40 wake probes on the spanwise planes
  z = -1.5, -0.5, 0.5, 1.5 (z-major; the observation is the plane average of u/U and v/U per 2D probe).
- `probes_wake__<case>.txt`: the 40 wake probes of the cylinder and square-cylinder cases.

On the released m-AIA grids the nominal probe points lie on, or within 2e-8 d of, cell faces. m-AIA samples the cell
that contains a probe, and which cell contains a point on a face depends on the face tie-break of the m-AIA version
(`getIdAtPoint`). Each coordinate within 1e-7 d of a face is therefore moved by 1e-6 d into the cell that the
released m-AIA selects, so that the probe -> cell mapping is the same in every m-AIA version.
`load_policy.load_probes(path)` reads a probe file.

## Files

All files are tracked in git; the largest is 2.8 MB. Files larger than 20 MB would go to `policies/_large/`, which is
listed in `.gitignore`. For the Kolmogorov policies, `kolmogorov_re40/*/seed30/params.msgpack` (role `weights_flax`)
holds the same weights in the format of the Flax trainers of `examples/jax/kolmogorov_gppo/`.
