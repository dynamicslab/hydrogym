# Changelog

Notable user-facing changes to HydroGym. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## 1.0.2

### Environments
- Gymnasium registration (`hydrogym.registration`): every environment of the catalogue with data is registered under its
  Hugging Face name (`hydrogym-maia/<name>-v0`, `hydrogym-nek/<name>-v0`, `hydrogym-jaxfluids/<name>-v0`); m-AIA environments
  other than `Cylinder_2D_Re200` need `probe_locations`.
- New NekRS backend (`hydrogym.nekrs`): channel and NACA0012 wing cases with an MPMD control interface.
- NACA0012 wing (Nek5000): case at Re_c = 2×10⁵ with the tripping of ref. (Wang, Atzori & Vinuesa 2025), 30 spanwise
  modes renewed every 0.008 c/U∞; time step 4×10⁻⁵ and 31 solver steps per action; the start state is the developed
  uncontrolled flow at t = 28 c/U∞. Updated Hugging Face case files and configuration of `NACA0012_3D_Re200000_AOA0`.
- NACA0012 wing (NekRS, `examples/nekrs/wing`): the defaults are the deployment of the Nek5000 case (time step, steps
  per action, scaling, trip), with the observations in the wall-tangent frame; opposition control with `wing_oc.yml`; the solver time step is
  set from the YAML (`environment.dt`) and checked at start-up. `deploy.py` samples the walls every 10 solver steps,
  also within the actions (`--wall-steps`; `0` for samples at the end of the actions), and `fig4d.py` takes C_d,f and
  C_d as time means over every solver step.
- NekRS Slurm examples run in the HydroGym container images; `examples/nekrs/gpu_share.sh` runs more NekRS ranks
  per node than GPUs.
- New and updated m-AIA cases: 2D and 3D open cavities (one and three jets), gust airfoil with a superposed gust
  field, and updated Hugging Face data for several existing cases.
- New JAX Kolmogorov preset for extreme-event control, with PPO and gradient-enhanced PPO example trainers.
- `Pinball_3D_Re150` now runs at Ma 0.1 by default (`max_control` 0.35, 352 lattice steps per action, start state a
  developed uncontrolled Ma 0.1 flow); the Ma 0.2 configuration is `Pinball_3D_Re150_Ma02`.
- Canonical gymnasium IDs for all gymnasium-API backends (`hydrogym.registration`, `gym.make(...)`).

### Solver interfaces and numerical robustness
- Shared helpers for external-process (MPMD) backends and Hugging Face data staging.
- Clearer errors for invalid configurations, mismatched MPMD launches, diverged flows and mismatched checkpoints,
  and for meshes or case files that are Git LFS pointers (a clone without `git lfs pull`).
- `hydrogym.__version__`.
- Faster m-AIA probe exchange when the actuation-power model is enabled.
- Firedrake environments with jets no longer recompile kernels at every step (much faster stepping).
- m-AIA MPMD launches: the application number can be set with `MAIA_MPMD_APPNUM` (e.g. for `srun --multi-prog`);
  otherwise `MPI_APPNUM` is used.
- m-AIA gust cases: the selected gust field file is installed atomically in the m-AIA working directory; the
  transverse gust can be selected as `gust='transverse'` (also `'fig2c'`).
- NekRS: an unknown `probes.frame` value is an error (it was treated as `global`).
- Nek5000: the workspace preparation no longer links a `CASE.sch` file shipped with an environment (Nek5000 writes
  it and does not start if it exists); `NekEnv` stops with a clear error when the reward has no `dUdy` reference.
- JAX channel: `step_env` no longer computes an unused gradient of the wall shear stress (faster steps; results
  unchanged).
- m-AIA `DRA2303Jet`, `ZPGTBLJet`: the jets no longer fall back to about the previous action in the first
  boundary-condition call of each action step, and `reset()` clears the jet state, so episodes no longer depend on
  the previous one.
- Nek5000 wing case: the start-up handshake works with any number of solver ranks, and the probe search covers only
  the probes of each rank.
- Hugging Face data: Nek5000 environments download only the restart set the case reads (`restart_files/`), not other
  restart sets stored next to it; a failed Hub lookup with a usable local fallback is logged as a warning; the cache
  link of an environment is created atomically, so concurrent runs with the same cache do not fail.
- Nek5000 examples: `prepare_workspace.py` takes the solver version (`--profile`) from the environment when it is
  not given; `gym_make_demo.py` takes `--local-dir` for offline use.
- JAX run scripts: with fewer than 4 CPUs they set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, which avoids a
  hang of the XLA CPU thread pool in the channel case.
- Firedrake, JAX and Nek fixes for substepping, start-up order and configuration handling; `gymnax` pinned to `<1`;
  `navix>=4` in the JAX extras (Python >= 3.12), imported only by the Navix wrapper.

### Actuation and power reporting
- The controlled environments report their actuation power in `info` and can add it to the reward
  (`power_weight`, default 0); the Nek5000 backend for its two wing cases (`drl/drl_power.f`), not for the channels.
  The m-AIA `DRA2303Jet` and `ZPGTBLJet` cases take the power of their jets from the solver (needs the pinned
  `third_party/m-AIA`).

### Pretrained policies
- `policies/` with trained policies for pinball, cylinder, square cylinder, cavity, gust airfoil and Kolmogorov
  cases, an index file and a numpy-only loader.
- Pretrained NACA0012 wing policies (`policies/naca0012_wing_re2e5`), run by `examples/nekrs/wing`
  (`--policy td3:<id>`). The Stable-Baselines3 file of the TD3 policy holds the weights and settings only (the
  learning-rate schedule is rebuilt from `learning_rate` when it is loaded).
- Examples that run and plot the released policies (`examples/jax/released_policies`, `examples/maia/released_policies`).

### Documentation, tests and CI
- Developer guides, installation pages for NekRS, reference solver templates, regenerated API pages.
- Extended unit tests; CI runs linting, the test suite and a docstring-coverage check.
- Container images with all backends: one NVIDIA GPU image (compute capabilities 7.5 to 12.0) and one CPU image,
  with build and test scripts; an Apptainer recipe for NVIDIA GH200 (aarch64).
- Docs: NekRS listed among the solver backends; a JAX-Fluids example page; citation of the Nature article.
- Nek5000 examples: opposition control takes the wall-normal component of each actuator's observation (`obs[1::2]`);
  the train-then-evaluate workflow evaluates in the training environment (one MPMD launch serves one `NekEnv`).
- m-AIA examples and docs: offline use with `snapshot_download(local_dir=...)` and `local_fallback_dir`.
- NekRS wing example: `trip_origin.txt` must hold the origin, time step, mode count and interval (run directories of
  earlier versions stop with a message); run names of the examples (`runs/zero_dt4e-5`, `runs/zero_ref1`).
- Firedrake cavity examples: `sine-forcing.py` restarts from the checkpoint of `unsteady.py`.
- JAX PPO example: PureJaxRL licence and attribution.
- Nek5000 cases: unused source variants, build products and private build paths removed; mesh map files in Git LFS.

### Behaviour changes that can change results
- `terminated` now means invalid physics only; reaching the step budget sets `truncated` (m-AIA, Nek).
- `Pinball_3D_Re150` defaults to Ma 0.1; use `Pinball_3D_Re150_Ma02` for the previous Ma 0.2 configuration.
- NACA0012 wing deployment: action amplitude in friction-velocity units of the training channel; observation scaling
  and probe height by the friction velocity of the uncontrolled wing at Re_c = 2×10⁵ (Nek5000 and NekRS).
- NACA0012 wing (Nek5000 and NekRS): the tangential velocity observed by the policies is the component along the
  wall tangent (`frame: wall`, default).
- NACA0012 wing (Nek5000): trip of 30 spanwise modes renewed every 0.008 c/U∞ (was 60 modes every 0.004), Re_c = 2×10⁵
  in the case file, time step 4×10⁻⁵ and 31 solver steps per action (Hugging Face configuration).
- m-AIA environments: `step(a)` returns the state at the end of the interval in which `a` acted.
- m-AIA environments: `reset()` re-initialises the solver source terms (e.g. the dynamic sponge of the 3D pinball), so
  a reset reproduces a fresh start (requires the pinned `third_party/m-AIA` version).
- Time-step count of the JAX spectral solvers corrected (whole steps per action).
- Nek5000 small wing (`small_wing`): the sensing plane uses the viscosity of its Re_c = 7.5×10⁴ (it was placed for
  Re_c = 10⁵, at y⁺ ≈ 11 instead of 15), and the probe search covers only the probes of each rank.
- JAX Kolmogorov control forcing and grid corrected, and the 2/3 dealiasing of the advection term (previously without
  effect) applied (legacy options `legacy_control_scaling`, `legacy_grid_endpoint`, `legacy_dealiasing`).
- JAX channel jets corrected (legacy option `legacy_jets`).
- Firedrake substep iteration count and BDF start-up order corrected (legacy options `legacy_substep_iteration`,
  `legacy_startup_order`); Firedrake `Cylinder` jets are zero-net-mass-flux (legacy option `legacy_jets`).
- Firedrake velocity, pressure and vorticity probe observations are returned in the order the probes are given and are
  identical on every MPI rank.
- Actuation limits of some m-AIA environments updated (see each environment's config).
- Some Hugging Face environment data replaced or removed (2D cavity, pinball checkpoints); an m-AIA build of the
  pinned `third_party/m-AIA` version is required for the updated m-AIA cases.

### Removed
- `hydrogym.distributed` (empty package).
