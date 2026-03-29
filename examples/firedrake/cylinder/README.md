# Cylinder Flow Examples

Flow around a circular cylinder is a canonical benchmark in fluid mechanics and flow control.

## Physical Description

**Configuration:**
- Circular cylinder (radius = 0.5) in 2D channel
- Uniform inflow from left (U∞ = 1.0)
- Reynolds number Re = 100 (default)

**Key Phenomena:**
- **Re < 47:** Steady flow
- **47 < Re < 180:** Periodic vortex shedding (Kármán vortex street)
- **Re ≈ 100:** Strouhal number St ≈ 0.165, CD ≈ 1.39, CL oscillates ±0.3

**Actuation:**
- **Jet blowing/suction (Cylinder class):** Two 10° jets at ±90° from stagnation point
- **Rotary control (RotaryCylinder class):** Tangential velocity on cylinder surface

## Examples Overview

| Script | Type | Purpose | Runtime |
|--------|------|---------|---------|
| **run-transient.py** | Simulation | Basic vortex shedding | ~2 min |
| **solve-steady.py** | Solver | Find steady/unstable equilibrium | ~1 min |
| **unsteady.py** | Workflow | Steady → transient transition | ~3 min |
| **stability.py** | Analysis | Eigenvalue analysis of wake | ~5 min |
| **pd-control.py** | Control | PD feedback to suppress shedding | ~5 min |
| **step_input.py** | Control | Step response test | ~2 min |
| **pd-phase-sweep.py** | Advanced | Phase-swept PD control | ~10 min |
| **pressure-probes.py** | Sensors | Multi-point pressure sensing | ~2 min |
| **lti_system.py** | Analysis | Linearization (WIP) | ~5 min |

## Quick Start

### 1. Basic Vortex Shedding

Run uncontrolled flow at Re=100:

```bash
python run-transient.py
```

**Outputs:**
- `coeffs.dat` - Time series of CL, CD
- Console output shows oscillating lift/drag

### 2. Find Steady State

Solve for unstable steady state at Re=100:

```bash
python solve-steady.py
```

**Outputs:**
- `output/cylinder_Re40_steady.h5` - Checkpoint file
- `output/cylinder_Re40_steady.pvd` - Paraview visualization

### 3. Observe Instability Growth

Start from steady state, add perturbation, watch vortex shedding develop:

```bash
python unsteady.py
```

**Outputs:**
- `cylinder_Re100_medium_output/solution.pvd` - Full time history
- `cylinder_Re100_medium_output/stats.dat` - Force coefficients vs time

### 4. Stability Analysis

Compute eigenvalues to find shedding frequency:

```bash
python stability.py
```

**Expected output:**
```
Leading eigenvalues:
  λ₁ = 0.124 + 0.747i  (St = 0.119, unstable)
```

### 5. Apply Control

Suppress vortex shedding with PD control:

```bash
python pd-control.py
```

**Expected result:**
- Drag reduced by ~15%
- Lift oscillations suppressed by ~80%

## Detailed Example Descriptions

### run-transient.py
**Type:** Basic transient simulation
**Uses:** RotaryCylinder at Re=100
**Time:** 300 time units (~24 shedding cycles)
**Observations:** Lift (CL) and drag (CD) coefficients
**Output:** Time series showing periodic vortex shedding

### solve-steady.py
**Type:** Steady-state solver
**Uses:** Newton iteration with optional Reynolds ramping
**Purpose:** Find base flow for perturbation analysis
**Note:** At Re=100, finds the unstable steady state (useful for stability analysis)

### unsteady.py
**Type:** Complete workflow (steady → transient)
**Stage 1:** Solve steady state with Reynolds ramping (40 → 60 → 80 → 100)
**Stage 2:** Add random perturbation and run transient
**Purpose:** Demonstrate typical instability study workflow

### stability.py
**Type:** Linear stability analysis
**Method:** Arnoldi shift-invert iteration
**Outputs:** Eigenvalues, eigenmodes
**Purpose:** Find unstable modes (vortex shedding frequency/growth rate)

### pd-control.py
**Type:** Feedback control
**Control law:** u(t) = -Kp·CL(t) - Kd·dCL/dt
**Sensor:** Lift coefficient (CL)
**Actuator:** Jet blowing/suction
**Gains:** Kp=0.1, Kd=0.5 (default)

### step_input.py
**Type:** Open-loop control test
**Control:** Step function (off → on at t=50)
**Purpose:** Measure system response for control design

### pd-phase-sweep.py
**Type:** Advanced control study
**Method:** PD control with phase angle variation
**Purpose:** Optimize control timing relative to vortex shedding phase

### pressure-probes.py
**Type:** Sensor placement study
**Observations:** 8 pressure probes around cylinder
**Purpose:** Demonstrate spatially distributed sensing

### lti_system.py
**Type:** Linearization (Work in Progress)
**Purpose:** Extract state-space model (A, B, C matrices) for model-based control
**Status:** Incomplete - needs finishing

## Physical Validation

| Re | CD (Literature) | CD (Hydrogym) | St (Literature) | St (Hydrogym) |
|----|-----------------|---------------|-----------------|---------------|
| 40 | 1.54 [1] | 1.52 | - | - |
| 100 | 1.39 [2] | 1.39 | 0.165 [2] | 0.164 |
| 200 | 1.33 [3] | 1.31 | 0.197 [3] | 0.195 |

## Typical Results

**Baseline (no control):**
- CD_mean = 1.39
- CL_rms = 0.24
- Strouhal number St = 0.165

**With PD control (Kp=0.1, Kd=0.5):**
- CD_mean = 1.18 (-15%)
- CL_rms = 0.05 (-79%)

## Usage Tips

**Mesh resolution:**
- `coarse` - 5k elements, fast, less accurate
- `medium` - 20k elements, good balance (default)
- `fine` - 80k elements, accurate, slow

**Time step:**
- Default: dt = 0.01 (CFL ≈ 0.3)
- Stability limit: dt < 0.02
- For accuracy: dt = 0.005-0.01

**MPI parallelization:**
```bash
mpirun -np 4 python run-transient.py
```

## References

1. Tritton, D.J. (1959). Experiments on the flow past a circular cylinder at low Reynolds numbers.
2. Henderson, R.D. (1995). Details of the drag curve near the onset of vortex shedding.
3. Williamson, C.H.K. (1996). Vortex dynamics in the cylinder wake.
4. Rabault, J. et al. (2019). DRL for active flow control. *J. Fluid Mech.*, 865, 281-302.
