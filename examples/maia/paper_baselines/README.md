# Paper baselines: PPO / DDPG / TD3 on the fluidic pinball (m-AIA)

Reruns of the HydroGym paper's algorithm benchmarks on the fluidic pinball with Stable-Baselines3:
Exp. 1 (`Pinball_2D_Re100`) and Exp. 2 (`Pinball_3D_Re150`). [`train_pinball.py`](train_pinball.py)
has the hyperparameters (paper values where given, picks from the paper's ranges otherwise),
the 8 x 5 probe grid, per-episode logging and a deterministic test rollout at the end.

On one MAIA workspace (see [`../getting_started`](../getting_started)):

```bash
mpirun -np 1 python train_pinball.py --env Pinball_2D_Re100 --algo TD3 --seed 0 --out td3_s0 \
    : -np 1 maia properties_run.toml
```

Outputs in `--out`: `episodes.csv` (episode, steps, mean reward, mean and last-20-step sum C_D,
wall time), `model_epN.zip` checkpoints, `model_final.zip`, `eval_rollout.json` (C_D, C_L,
actions of the 200-step deterministic rollout). A 2D episode takes ~1 min with one m-AIA rank
on an RTX A6000 (500 episodes: ~9 h, more on a shared GPU), so for wall-time-limited jobs
`--max-hours` stops a run at an episode end and exits with code 3; started again with the same
`--out`, it resumes (`model_latest.zip`, `replay_buffer_latest.pkl`, `state.json`).

## AMD MI300A (AAC cluster, apptainer)

m-AIA comes from the ROCm image `docker.io/clagemann/hydrogym-rocm-6.3.3:v0.1` (hipstdpar build
for gfx942, module `MAIA/1.0-HIPCC-7.1-OMPI-5.0.5-SystemROCm-6.3.3`); the Python side is a venv
on the shared file system built inside that image. Paths are set in [`aac_env.sh`](aac_env.sh)
(`REPO`, `BASE`, ...).

```bash
# login node (network): image, venv, HF case download into $BASE/hf_cache and $BASE/cases
cd ~/hydrogym/examples/maia/paper_baselines
./setup_aac.sh
# 3 runs on one node, one APU each; resubmits itself every ~7.5 h until all are done
RUNS="PPO:0 DDPG:0 TD3:0" ENV=Pinball_2D_Re100 sbatch run_aac.sbatch
RUNS="PPO:0 DDPG:0 TD3:0" ENV=Pinball_3D_Re150 sbatch run_aac.sbatch
```

Runs go to `$BASE/runs/<ENV>/<algo>_s<seed>/` (case workspace, `train-<jobid>.log`, results in
`out/`). `HSA_XNACK=1` is set because the hipstdpar build relies on unified memory for host
allocations. If m-AIA hangs or faults on the cluster's newer amdgpu driver (as the ROCm 6.4
NekRS image did), the image's ROCm 6.3 user space is the first suspect.
