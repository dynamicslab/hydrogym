---
sidebar_label: cavity
title: hydrogym.maia.envs.cavity
---

Cavity Flow Environment
========================

This module provides cavity flow CFD environments for reinforcement learning,
designed for flow control applications in cavity configurations.

Hugging Face cases `Cavity_2D_Re{4200,7500}` (one jet) and `Cavity3Jet_2D_Re{4200,7500}` (three jets), on the
literature configuration of the open cavity (Barbagallo et al. 2009 / Sipp & Lebedev 2007), validated against the
finite-volume reference:

* domain x in [-1, 4.7] x [-1, 0.5] in case coordinates (= literature coordinates + 1.2; cavity [1.2, 2.2] x [-1, 0]),
1 D sponge layers outside the literature domain, slip top, slip floor upstream of x = 0.8 and downstream of x = 2.95,
velocity inlet, non-reflecting (plane-wave characteristic) outlet;
* D2Q9 cumulant collision, Ma 0.0125 (U_inf = 0.0072169 lattice units), uniform grid D/240;
* `Re4200` is run at solver Re 4070, the same distance above this grid's critical Reynolds number as Re 4200 is
above the reference Re_c = 4140 (weakly unstable); `Re7500` is the limit-cycle case;
* jets (equilibrium jet boundary condition, parabolic profile over 0.02): leading edge x = 1.175 on y = 0
(wall-normal), and for `Cavity3Jet` also the trailing cavity wall (x = 2.2, y = -0.25) and the leading cavity
wall (x = 1.2, y = -0.5); actions in [-1, 1] times `max_control` = centre velocity 0.35 U (one jet) /
0.69 U (three jets); one action = 633 LB steps = 0.019 D/U;
* restart = a developed uncontrolled state; `baseline_state.feather` = the uncontrolled time mean (40 D/U) at
every cell centre.

Actuation power (`info['actuation_power']` etc., `hydrogym.maia.wall_jets`): one actuator per jet,
`C_P = 2 P / (U^3 D)` per unit span (`D` = cavity depth, `referenceLength`; there is no drag coefficient).
The jet cells are the fluid cells next to the slot (exactly the cells of BC 2007); the jets are not
zero-net-mass-flux, `p_ref` is the mean pressure over the jets (one group), so the signed power is gauge-free but
leaves out the work of supplying the net flow. In this configuration (walls on cell faces, equilibrium jet BC) the
mass flux of the model agrees with a control-volume balance of the solver's flow within 2 %. The slot is 4.8 cells
wide, so the exit pressure is not resolved: the signed power (a small residual for a single jet, about 4e-4 of it
kinetic) changes by a factor 2-3 with the wall stencil; only its order of magnitude is meaningful. Cases whose walls
are off the cell faces are modelled link by link.

Timing of the MPMD loop: `step(a_k)` sets `a_k`, advances one control interval with it and returns the probe state
at the END of that interval (gym convention; `hydrogym.maia.env_core.MaiaFlowEnv.step`). `reset()` returns the
state of the restart after the reinitialisation.

3D cases `Cavity_3D_Re{4200,7500}_lz024` (one jet) and `Cavity3Jet_3D_Re{4200,7500}_lz024` (three jets)
-------------------------------------------------------------------------------------------------------------------
The same classes load the 3D cavities (`nDim = 3` in the property file):

* The `_lz024` cases are the literature cavity above extruded over a spanwise-periodic width L_z = 0.24 D
(48 planes), uniform grid D/200 (7.39 M cells), D3Q27 cumulant collision, Ma 0.0125, the plane-wave
characteristic outlet, the same sponge, jets and Reynolds numbers as 2D (`Re4200` at solver Re 4070). They are
the environments of the 3D cavity policies in `policies/`.
* **Literature coordinates**, unlike the 2D cases: cavity [0, 1] x [-1, 0], z in [-0.12, 0.12] (the 2D cases use
literature + 1.2 in x). Jets: leading edge x = -0.025 on y = 0; `Cavity3Jet` also trailing wall (x = 1,
y = -0.25) and leading wall (x = 0, y = -0.5); the slots span the full width. `max_control` as 2D (0.35 U /
0.69 U); one action = 527 LB steps = 0.019 D/U.
* **Probes are passed by the user** (`probe_locations` is required, flattened `[x0, y0, z0, x1, ...]`).
`probes_8planes.txt` in each HF folder holds the 1760 cells (8 spanwise planes x 220 cells) that the released 3D
policies observe; `observation_weights.json` gives their observation operator (50 shear-layer probes and the
sensor per plane, averaged over the planes).  `baseline_state.feather` (uncontrolled time mean over t = 20-60 D/U,
spanwise-averaged) is given at all cells of the 48 planes; the env takes the nearest
point.
* **Launch:** MPMD, 1 Python rank + N m-AIA ranks (4 GPUs: about 2.5 s per action). With `srun --multi-prog` set
`MAIA_MPMD_APPNUM` (0 for Python, 1 for m-AIA); `mpirun -np 1 python ... : -np 4 maia properties_run.toml`
sets `MPI_APPNUM` itself.  Needs an m-AIA version with the D3Q27 characteristic outflow (bc 4046); without it
bc 4046 silently falls back to its LODI default, i.e. a different case.
* **Episode length:** `max_episode_steps` is 200 (3.8 D/U), as in 2D. The controlled 3D flow needs far longer: a
typical evaluation is a roll-out of 60 D/U = 3156 actions with the fluctuation averaged over t = 40-60 D/U (the
transient lasts 16-40 D/U depending on the start). Pass
`max_episode_steps=3156` (or larger), or chain episodes without `reset()`.
* **Start state:** `restart_.Netcdf` is a developed uncontrolled m-AIA state (t = 160 D/U).

## CavityBase Objects

```python
class CavityBase(MaiaFlowEnv)
```

Base class for cavity flow environments with Hugging Face integration.

This environment implements two reward strategies:
- 'baseline_mean': Penalize deviation from a pre-computed baseline state
- 'running_mean': Penalize deviation from a running average

**Attributes**:

- `reward_strategy` - Strategy for reward computation ('baseline_mean' or 'running_mean').

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cavity base environment.

**Arguments**:

- `env_config` - Environment configuration dictionary containing:
  - reward_strategy (str): 'baseline_mean' or 'running_mean'. Default: 'baseline_mean'

#### get\_reward

```python
def get_reward() -> Tuple[float, Dict]
```

Compute the reward based on the selected strategy.

For 'running_mean': Penalizes deviation from exponential moving average.
For 'baseline_mean': Penalizes deviation from pre-computed baseline.

**Returns**:

  Tuple containing:
  - reward: Negative sum of absolute deviations
  - obj_dict: Empty dictionary (for compatibility)
  

**Raises**:

- `ValueError` - If unknown reward strategy is specified.

## Cavity Objects

```python
class Cavity(CavityBase)
```

Single-jet cavity environment.

This environment simulates cavity flow with a single jet actuator
for flow control.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the cavity environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Convert RL action to CFD actuation format.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

## Cavity3Jet Objects

```python
class Cavity3Jet(CavityBase)
```

Three-jet cavity environment.

This environment simulates cavity flow with three independent
jet actuators for flow control.

**Attributes**:

- `numJetsInSimulation` - Number of jet actuators in the CFD simulation.

#### \_\_init\_\_

```python
def __init__(env_config: Dict)
```

Initialize the 3-jet cavity environment.

**Arguments**:

- `env_config` - Environment configuration dictionary.

#### convert\_action

```python
def convert_action(action: np.ndarray) -> np.ndarray
```

Convert RL action to CFD actuation format.

**Arguments**:

- `action` - Action array from the RL agent.
  

**Returns**:

  Action sequence for the CFD solver.

