# Released policies in m-AIA: run and plot

Run one released policy of `policies/` (ids in `policies/MANIFEST.yaml`) against the uncontrolled flow in its
HydroGym m-AIA environment, and plot the roll-outs in the style of the figures of Lagemann et al., *Nature* 657, 369–376 (2026), [doi:10.1038/s41586-026-10917-6](https://doi.org/10.1038/s41586-026-10917-6).
Each figure shows exactly one policy and the uncontrolled run from the same start state.

| file | what it does |
|---|---|
| `prepare_run.py` | creates the run directory of a policy (environment data, the policy's settings); no MPI |
| `run_policy.py` | MPMD application: uncontrolled roll-out, `env.reset()`, controlled roll-out; makes the figures |
| `plot_policy.py` | the figures from a stored `run.npz` (numpy + matplotlib only) |
| `run_example.sh` | the three steps in one command |
| `policy_cases.py` | per-case settings: environment, configuration, observation wiring, paper quantity |

The Kolmogorov policies run in the HydroGym JAX solver: see `examples/jax/released_policies/`.

## Requirements

- HydroGym with the m-AIA backend (`pip install hydrogym[maia]`) and an m-AIA executable with MPMD support (see
  `examples/maia/getting_started/`);
- an MPI launcher for MPMD jobs (`mpirun -np 1 python ... : -np N maia properties_run.toml`);
- matplotlib for the figures (or `--no-plot --store_data`, and plot `run.npz` on another machine).

The HydroGym container images have all of this (`clagemann/hydrogym-nvhpc-26.5_cuda-12.9:1.0.2` for NVIDIA GPUs,
`clagemann/hydrogym-cpu:1.0.2`; the examples are in `/opt/hydrogym/examples`, which is not writable for a non-root
user: copy them first, `cp -r /opt/hydrogym/examples ~/hydrogym-examples`). In the GPU image `maia` starts the
m-AIA build for the visible GPU. On a machine with fewer cores than MPI ranks, add `--oversubscribe`
(`MPIRUN="mpirun --bind-to none --oversubscribe"`).

The environment data is downloaded from the Hugging Face Hub (`dynamicslab/HydroGym-environments`) on first use. On
machines without internet access, download the environment folder elsewhere and pass it with `--env-dir`
(`PREPARE_ARGS="--env-dir /path/to/<Environment>"` for `run_example.sh`).

## Quick start

```bash
cd examples/maia/released_policies
./run_example.sh cylinder_2d_re200/ppo/seed1 runs/cylinder_re200 1
# -> runs/cylinder_re200/run.png
./run_example.sh cylinder_2d_re200/ppo/seed1 runs/cylinder_re200 1 --readable --store_data
# -> also run_readable.png, run_actions_readable.png, run.npz, policy_run.json
```

`run_example.sh <policy-id> <work-dir> [m-AIA ranks, default 1] [run_policy.py options]` calls

```bash
python prepare_run.py --policy cylinder_2d_re200/ppo/seed1 --work-dir runs/cylinder_re200
cd runs/cylinder_re200
mpirun -np 1 python ../../run_policy.py --policy cylinder_2d_re200/ppo/seed1 --out run.npz \
    : -np 1 maia properties_run.toml
```

All options after the m-AIA ranks (including `--readable` and `--store_data`) go to `run_policy.py`, which makes the
figures at the end of the run; `--file-role` also goes to `prepare_run.py`. Set `MPIRUN` (e.g.
`"mpirun --bind-to none"`), `MAIA` (path of the m-AIA executable) and `PYTHON` if needed; `PLOT=0` makes no figures
and stores `run.npz` instead (`--no-plot --store_data`), e.g. on a cluster without matplotlib.

## One command per case

```bash
# 2D fluidic pinball, Re 100 (ppo_tuned, ppo_IM: max_control 0.3125, the default of Pinball_2D_Re100)
./run_example.sh pinball_2d_re100/ppo_tuned/seed0 runs/pinball_re100_tuned 1
./run_example.sh pinball_2d_re100/ppo_IM/seed0   runs/pinball_re100_IM 1
./run_example.sh pinball_2d_re100/ppo_tuned/seed0 runs/pinball_re100_tuned_6 1 --start 6   # other start state
./run_example.sh pinball_2d_re100/ppo_power/seed0 runs/pinball_re100_power0 1   # max_control 0.25

# 2D fluidic pinball, Re 150
./run_example.sh pinball_2d_re150/ppo/seed2       runs/pinball_re150_ppo 1
./run_example.sh pinball_2d_re150/td3_long/seed2  runs/pinball_re150_td3 1
./run_example.sh pinball_2d_re150/ppo_power/seed1 runs/pinball_re150_power1 1

# 2D cylinder (Re 200, 1000, 3900) and square cylinder (Re 200; control from action 67): 100 actions each
./run_example.sh cylinder_2d_re200/ppo/seed1                                 runs/cylinder_re200 1
./run_example.sh cylinder_2d_re1000/ppo/seed2                                runs/cylinder_re1000 1
./run_example.sh cylinder_2d_re1000/ppo_finetune_from_re200/seed2            runs/cylinder_re1000_ft 1
./run_example.sh cylinder_2d_re3900/ppo/seed1                                runs/cylinder_re3900 1
./run_example.sh cylinder_2d_re3900/ppo_finetune_from_re1000/seed2           runs/cylinder_re3900_ft 1
./run_example.sh square_cylinder_2d_re200/ppo/seed0                          runs/square_re200 1
./run_example.sh square_cylinder_2d_re200/ppo_finetune_from_cylinder_re200/seed2 runs/square_re200_ft 1

# NACA0012 gust, Re 1000, alpha 20 deg, 3D (gust fig2c)
./run_example.sh naca0012_gust_re1000/td3_3jet_jet1.5U/seed1 runs/gust3d_td3 4

# 2D open cavity, Re 4200 (one jet) and Re 7500 (one and three jets)
./run_example.sh cavity_2d_re4200/ppo_tuned/seed0 runs/cavity_re4200_tuned 1
./run_example.sh cavity_2d_re4200/ppo_IM/seed0   runs/cavity_re4200_IM 1
./run_example.sh cavity_2d_re4200/ppo_tuned/seed0 runs/cavity_re4200_tuned_103 1 --start 103   # other start state
./run_example.sh cavity_2d_re7500/ppo_1jet/seed0 runs/cavity_re7500_1jet 1
./run_example.sh cavity_2d_re7500/ppo_3jet/seed1 runs/cavity_re7500_3jet 1

# 3D cases (GPU m-AIA build; 4 GPUs: 1 Python rank + 4 m-AIA ranks)
# 3D pinball Re 150, default configuration (Ma 0.1): controlled and uncontrolled roll-out as separate runs
./run_example.sh pinball_3d_re150/ppo_tuned/seed0 runs/pinball3d_re150_tuned 4 --runs ctl
./run_example.sh pinball_3d_re150/ppo_tuned/seed0 runs/pinball3d_re150_tuned_unc 4 --runs unc
python plot_policy.py runs/pinball3d_re150_tuned/run.npz runs/pinball3d_re150_tuned_unc/run.npz
./run_example.sh pinball_3d_re150/ppo_IM/seed0 runs/pinball3d_re150_IM 4 --runs ctl
./run_example.sh pinball_3d_re150/ppo_IM/seed0 runs/pinball3d_re150_IM_unc 4 --runs unc
python plot_policy.py runs/pinball3d_re150_IM/run.npz runs/pinball3d_re150_IM_unc/run.npz
./run_example.sh cavity_3d_re7500/ppo_IM_3jet/seed3 runs/cavity3d_re7500 4
```

On a cluster, run `prepare_run.py` on a login node and put the `mpirun` line into the job script.

m-AIA ranks: the 2D commands use one m-AIA rank (`<n-maia-ranks>` = 1, the default of `run_example.sh`). In the GPU
image every m-AIA rank of `run_example.sh` runs on the first visible GPU, and several ranks sharing one GPU are much
slower than one rank (by a factor of 40 or more); the results agree to round-off. Use one m-AIA rank per GPU. With the CPU build (the CPU
image), use one rank per core instead, e.g. `./run_example.sh cylinder_2d_re200/ppo/seed1 runs/cylinder_re200 4`.
The 3D cases need several GPUs (one m-AIA rank per GPU, see "m-AIA ranks and GPUs" in the container docs).

3D cavity: the default roll-out of `cavity_3d_re7500` (3156 actions, 60 D/U) needs about 2.6 s per action on four
NVIDIA GH200 GPUs, so about 4.5 h for the uncontrolled and the controlled roll-out; `--actions 1052` (20 D/U) about
1.5 h for both. `cavity_3d_re4200` runs 1052 actions by default.

3D gust (`PREPARE_ARGS="--dim 3"`): m-AIA needs about 410 GB of host memory on its 4 ranks, and the job needs about
450 GB on one node, more than a node with 480 GB can reliably provide. Use two nodes with two m-AIA ranks each (about
225 GB per node). With Slurm and the
Apptainer image, start every rank in its own container with `srun --multi-prog` (without `--cleanenv`: the ranks need
the `PMIX_*`/`SLURM_*` variables; `srun --mpi=list` names the PMIx plugin, e.g. `pmix` or `pspmix`; add the `--bind`
and `--home` options your site needs to the `apptainer exec` lines):

```bash
# on a login node: python prepare_run.py --policy naca0012_gust_re1000/td3_3jet_jet1.5U/seed1 --work-dir runs/gust3d --dim 3
# in the job (#SBATCH --nodes=2 --gpus-per-node=4 --exclusive --mem=0), from runs/gust3d:
cat > maia_rank.sh << 'EOF'
#!/bin/bash
# one GPU per m-AIA rank; node 0 also holds the Python rank (local task 0), so its m-AIA ranks use GPUs 0 and 1
off=0; [ "$SLURM_NODEID" = 0 ] && off=1
export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID - off)) MAIA_MPMD_APPNUM=1
exec apptainer exec --nv /path/to/hydrogym-gh200.sif maia properties_run.toml
EOF
chmod +x maia_rank.sh
printf '%s\n' \
  "0 env MAIA_MPMD_APPNUM=0 apptainer exec --nv /path/to/hydrogym-gh200.sif python /path/to/released_policies/run_policy.py --policy naca0012_gust_re1000/td3_3jet_jet1.5U/seed1 --out run.npz" \
  "1-4 ./maia_rank.sh" > mpmd.conf
srun --mpi=pmix -N 2 -n 5 --ntasks-per-node=3 --distribution=block -c 8 --cpu-bind=cores --multi-prog mpmd.conf
```

Both roll-outs take about 50 min on 2 nodes with 4 NVIDIA GH200 GPUs each; the result does not depend on the number of
nodes (agreement to round-off).

3D pinball: run the uncontrolled and the controlled roll-out as two separate runs (`--runs unc`, `--runs ctl`) and plot
both files together, so that each roll-out starts from the restart state of the case. A run with one roll-out only
always stores its `run.npz` and `policy_run.json` (also without `--store_data`; it prints one line saying so) and
makes no figures; `plot_policy.py ctl/run.npz unc/run.npz` then plots both parts (add `--readable` for the readable
figures).

## Options of `run_policy.py`

| option | default | meaning |
|---|---|---|
| `--policy` | (required) | policy id of `policies/MANIFEST.yaml` (the run directory must be prepared for it) |
| `--actions N` | case default | controlled actions after the start (`n_actions` in `policy_cases.py`; 2D cylinders, square cylinder and 3D pinball: 100) |
| `--start K` | 0 (square cylinder: 67) | uncontrolled actions before the policy is switched on (other start states) |
| `--runs` | `unc,ctl` | roll-outs: `unc` (zero action), `ctl` (policy), or both; one part only stores its data |
| `--readable` | off | also the readable figures: `run_readable.png` and the actions, `run_actions_readable.png` |
| `--store_data` | off | also write `run.npz` and `policy_run.json` |
| `--no-plot` | off | no figures (with `--store_data`; plot `run.npz` elsewhere) |
| `--out` | `run.npz` | name of the npz in the run directory; its name without `.npz` is the prefix of the figures |
| `--file-role` | loader file | another weights file of the entry (its `role` in the `files` list of the index) |

`run_policy.py` builds the environment with the probe locations the policy observes and `obs_normalization_strategy`
`"none"`, and maps the probe data to the policy input as in `policy_cases.py` (`OBSERVATIONS`):

| case | observation | action |
|---|---|---|
| pinball 2D | u/U, v/U at the 40 wake probes of `policies/probes/` (+ previous action) | three rotation rates |
| pinball 2D, `ppo_power` | the same + previous action + C_D, C_L, C_P of each cylinder (see below) | three rotation rates |
| pinball 3D | the same, averaged over 4 spanwise planes (`pinball_3d_wake_probes.json`) | 3D command `clip(1.6 a, -1, 1)` |
| cylinder, square cylinder | u/U, v/U at the 40 wake probes of the entry's `probes_file` | jet(s) |
| NACA0012 gust | u/U, v/U at the 40 probes of the controller file (3D: mean over 8 spanwise planes) | three jets |
| cavity 2D | the controller's 224 points, reduced by its own observation operator | jet(s), rate-limited |
| cavity 3D | `probes_8planes.txt` of the environment, per plane via `observation_weights.json` | jet(s), rate-limited |

The environment settings of each policy entry are applied in `environment_config.yaml` of the run directory
(`policy_cases.CASES[...]["config"]`); those that differ from the environment's default configuration are, e.g.,
`max_control` 0.25 for the 2D pinball `ppo_power` controller at Re 100 (see below), the jet bound and action
interval of the gust controller file, 88 lattice steps per action and `max_control` 0.0173 (0.3 U) for the cylinder
Re 200. The gust policies run with the gust `fig2c` (`env_config["gust"] = "fig2c"`).

The rotation bound of the 2D pinball at Re 100: `Pinball_2D_Re100` defaults to `max_control` 0.3125 (a cylinder
surface speed of 2.71 U per unit action), the bound of `ppo_tuned` and `ppo_IM`, which their entries also state. The
`ppo_power` controllers run with `max_control` 0.25 (2.17 U) at Re 100 and 0.4 with 152 steps at Re 150. They also
observe the drag and lift coefficients of each cylinder, the means of the per-step wall forces that m-AIA has
written since the previous action (`calculateWallForces` with interval 1 in `properties_run.toml` of the run
directory), and the actuation power per cylinder of the environment (`info["actuation_power_per_actuator"]`).
Controllers with `hold_actions` > 0 hold their base rotation for that many actions before the controller acts; the
default roll-out adds these actions to `n_actions` (600), and the summary window is the second half of the actions
after them.

## Evaluation protocol

The default roll-out and the summary window of each case (`policy_cases.py`; actions are policy actions, counted from
control onset). The drag summary is the mean $C_D$ of the controlled run over the window against the mean $C_D$ of the
uncontrolled run over the same window.

| Case | Roll-out (actions) | Summary window | Reported quantity |
|---|---|---|---|
| Pinball 2D Re 100 / Re 150 | 200 / 600 | 100-200 / 300-600 | mean $C_D$ vs uncontrolled |
| Pinball 2D, `ppo_power` | 600 after the hold phase | 300-600 after the hold | mean $C_D$, $C_P$ and net saving |
| Pinball 3D Re 150 | 100 | 50-100 | mean $C_D$ vs uncontrolled |
| Cylinder 2D Re 200 / 1000 / 3900 | 100 | 50-100 | mean $C_D$ vs uncontrolled |
| Square cylinder Re 200 | 100 (control starts at action 67 of the restart) | 50-100 | mean $C_D$ vs uncontrolled |
| NACA0012 gust, Re 1000 | 200 | whole run | mean $C_L$; the figure shows the gust passage (2.53-5.0 c/U) |
| Cavity 2D Re 4200 | 237 (about 4.5 D/U) | whole run from sample 37 | TKE (1 D/U trailing window) / mean uncontrolled TKE; ratio at action 237 |
| Cavity 2D Re 7500 | 200 | whole run | the same, without the skipped samples |
| Cavity 3D Re 7500 / Re 4200 | 3156 (60 D/U) / 1052 (about 20 D/U) | whole run | TKE (2 D/U trailing window) / mean uncontrolled TKE |

Every roll-out starts from the same stored restart state of the environment: `reset(seed=...)` does not change the
flow, so the runs test the policy on its training start state. The first 37 samples of the cavity Re 4200 TKE are left
out of the figure and the normalisation, because controlled and uncontrolled TKE still coincide there. The complete
time series of a run are in `run.npz` (`--store_data`).

## Output

By default `run_policy.py` writes one figure (PNG) into the run directory:

- `run.png`: the paper quantity, policy and uncontrolled run, in the paper style (the panel size, 2.2559 x 0.9331
  in, fonts and colours as in the paper) with tick labels, axis labels and a small legend.

`--readable` adds

- `run_readable.png`: a larger version with the policy, environment and summary in the title;
- `run_actions_readable.png`: the actions of the policy (several actuators in shades of the policy colour; the
  uncontrolled run has zero action).

The y-range is the case's range (`policy_cases.py`), extended where the data does not fit in it (e.g. a drag
overshoot after control onset). The summary of the run is printed at the end; it is always computed over the whole
run (drag: mean over the case's summary window, e.g. actions 50-99 of the 100 actions of the 2D cylinders, the square
cylinder and the 3D pinball, counted from 0 at control onset).

Time axis: convective time from control onset, except for

- the gust: `run.png` shows an excerpt of the gust passage, from the rise of the lift to the time the tail of the
  gust reaches the leading edge (`excerpt` in `policy_cases.py`, in convective time after the reset; the matching
  actions follow from the action interval of the 2D and the 3D case), mapped onto a normalised time axis 0-1. The
  roll-out is the whole default run; the readable figures show all of it in convective time;
- the cavity: the TKE starts one averaging window after control onset, so the time of `run.png` and
  `run_readable.png` is normalised to 0-1 over the plotted data (0 = first, 1 = last point). The TKE is divided by the
  mean uncontrolled TKE over the plotted samples (1 = the mean of the uncontrolled curve shown), and the summary's
  `episode_end_ratio` is the controlled TKE at the end of action 200 (counted from control onset) divided by the
  same mean. The cavity Re 4200 figures start 37 TKE samples (0.7 D/U) later (`tke_skip` in `policy_cases.py`): the
  policies' first actions are tiny, and there the controlled and uncontrolled TKE coincide. Its default roll-out is
  therefore 237 actions instead of 200, so that the plotted part is as long as before; its `episode_end_ratio` is
  taken at the end of action 237 (`episode_end` in `policy_cases.py`), the last point of the figure, and normalised
  over the plotted samples 37-237.

`--store_data` also writes

- `run.npz`: per action and roll-out (`unc_*`, `ctl_*`): `action` (normalised), `command` (solver units),
  `C_D_body`, `C_L_body` (force coefficients of every body at the end of the action interval, normalised as in the
  environment's reward), `C_D`, `C_L` (sums over the bodies), `C_P` (actuation power of the environment's power
  model, NaN where there is none) and, for the cavity, `O` (the shear-layer observation, from the state at the start
  on); `start`, `n_actions`, `dt_action` (convective time units per action) the protocol; `meta` (JSON) the policy
  id, environment, dimension, `n_actions`, `dt_action` and the physical settings (`max_control`, lattice steps per
  action `substeps`, `Ma`, `Re`);
- `policy_run.json`: the policy id, environment, dimension and the settings of the run that are set (environment
  settings that differ from the environment's configuration, e.g. `max_control`, the gust option, `actions`,
  `start`, `runs`, `file_role`).

The figure settings (quantity, axis labels and limits, summary window, label and colour of the policy) come from
`policy_cases.py` and the policy id, not from the npz. `plot_policy.py` makes the figures again from stored files:

```bash
python plot_policy.py run.npz [--readable] [--out PREFIX] [--xlim A B] [--ylim A B] [--max-action N]
```

It writes `PREFIX.png` (default `PREFIX`: the first npz without `.npz`, e.g. `run.png`) and, with `--readable`,
`PREFIX_readable.png` and `PREFIX_actions_readable.png`. `--max-action N`: only the first N controlled actions (for
the gust instead of its excerpt); the summary numbers stay those of the whole run. `--xlim A B` is in the units of
the time axis (0-1 where it is normalised); `--ylim A B` sets the y-range exactly. Several files of one policy, e.g.
the two parts of a split run, are merged.

The y-axis is the paper's quantity: drag coefficient (pinball: sum over the three cylinders; cylinders), lift
coefficient (gust), or TKE / mean uncontrolled TKE (cavity: the fluctuation energy of the shear-layer observation,
mean over its components of the variance over a trailing window of 1 D/U in 2D and 2 D/U in 3D, divided by the mean
of the uncontrolled run).

Colours: uncontrolled `#2E2E2E`; PPO `#1f77b4`, GPPO `#2ca02c`, TD3 `#9467bd`, DDPG `#d62728` (from the variant of
the policy id); other controllers `#8C564B`.

`python ../../../policies/load_policy.py <id>` prints the interface of an entry (format, actions, observation size,
`max_control`).
