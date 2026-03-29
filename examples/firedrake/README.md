# Firedrake Examples

Comprehensive examples demonstrating Hydrogym's Firedrake backend for CFD and flow control.

## Directory Structure

```
firedrake/
├── README.md                   # This file
├── env_test/                   # Environment testing scripts
│   ├── test_firedrake_env.py  # Comprehensive test suite
│   └── run_example_docker.sh  # Docker runner script
├── cavity/                     # Lid-driven cavity flow
│   ├── README.md               # Cavity-specific documentation
│   ├── run-transient.py        # Basic turbulent simulation
│   ├── solve-steady.py         # Steady-state solver
│   ├── unsteady.py             # Steady → turbulent workflow
│   ├── stability.py            # Eigenvalue analysis
│   └── sine-forcing.py         # Sinusoidal forcing
├── cylinder/                   # Flow around circular cylinder
│   ├── README.md               # Cylinder-specific documentation
│   ├── run-transient.py        # Basic vortex shedding
│   ├── solve-steady.py         # Steady/unstable equilibrium
│   ├── unsteady.py             # Steady → shedding workflow
│   ├── stability.py            # Wake stability analysis
│   ├── pd-control.py           # PD feedback control
│   ├── step_input.py           # Step response test
│   ├── pd-phase-sweep.py       # Phase-swept control
│   ├── pressure-probes.py      # Multi-point sensing
│   └── lti_system.py           # Linearization (WIP)
├── pinball/                    # Three-cylinder configuration
│   ├── README.md               # Pinball-specific documentation
│   ├── run-transient.py        # Complex three-body wake
│   ├── solve-steady.py         # Steady equilibrium
│   └── unsteady.py             # Wake dynamics workflow
└── step/                       # Backward-facing step
    ├── README.md               # Step-specific documentation
    ├── run-transient.py        # Separated flow simulation
    ├── solve-steady.py         # Steady recirculation zone
    ├── unsteady.py             # Separation → shedding
    └── step-control.py         # Step input control
```

## Quick Start

### 1. Test Your Installation

First, verify your Firedrake installation works:

```bash
cd env_test
python test_firedrake_env.py --environment cylinder --num-steps 10
```

This should run a short simulation and output force coefficients.

### 2. Run Your First Example

Try the simplest example - cylinder vortex shedding:

```bash
cd cylinder
python run-transient.py
```

You should see output like:
```
t: 1.00,   CL: -0.234,   CD: 1.387   Mem: 15.2
t: 2.00,   CL: 0.287,    CD: 1.402   Mem: 15.3
...
```

### 3. Explore Other Environments

Each environment has similar basic examples:

```bash
# Lid-driven cavity turbulence
cd cavity && python run-transient.py

# Three-cylinder wake
cd pinball && python run-transient.py

# Backward-facing step
cd step && python run-transient.py
```

## Example Types

All environments follow a standardized structure:

### Tier 1: Essential (All environments have these)

**run-transient.py** - Basic time-stepping simulation
- Good starting point
- Shows typical output
- Runtime: 2-5 minutes

**solve-steady.py** - Steady-state solver
- Newton iteration with Reynolds ramping
- Finds equilibrium states
- Useful for perturbation studies

**unsteady.py** - Complete workflow
- Stage 1: Solve steady state
- Stage 2: Perturb and run transient
- Demonstrates typical research workflow

### Tier 2: Advanced (Selected environments)

**stability.py** - Linear stability analysis
- Available for: Cavity, Cylinder
- Computes eigenvalues and modes
- Finds instability frequencies

**Control examples** - Feedback/feedforward control
- Cavity: `sine-forcing.py`
- Cylinder: `pd-control.py`, `step_input.py`
- Step: `step-control.py`

### Tier 3: Specialized (Domain-specific)

- `cylinder/pd-phase-sweep.py` - Phase-amplitude control
- `cylinder/pressure-probes.py` - Multi-point sensing
- `cylinder/lti_system.py` - Linearization (incomplete)

## Environments Overview

| Environment | Description | Re (default) | Key Phenomena |
|-------------|-------------|--------------|---------------|
| **Cavity** | Lid-driven cavity | 7500 | Recirculation, turbulence |
| **Cylinder** | Circular cylinder | 100 | Vortex shedding, St=0.165 |
| **Pinball** | Three cylinders | 100 | Complex wake, mode switching |
| **Step** | Backward step | 600 | Separation, reattachment |

## Typical Workflow

For research or control design, follow this progression:

```bash
# 1. Basic simulation - understand the flow
python run-transient.py

# 2. Find steady state - get base flow
python solve-steady.py

# 3. Full workflow - see instability develop
python unsteady.py

# 4. Stability analysis - find unstable modes (if available)
python stability.py

# 5. Apply control - test control strategies (if available)
python pd-control.py
```

## Running Examples

### Basic Usage

```bash
python run-transient.py
```

### With MPI Parallelism

```bash
mpirun -np 4 python run-transient.py
```

### In Docker

```bash
cd env_test
bash run_example_docker.sh
```

## Output Files

Examples typically generate:

- **`.pvd` files** - Paraview visualization
  - Open with: `paraview solution.pvd`
- **`.h5` files** - Checkpoint files for restart
  - Use with: `flow = hgym.Cylinder(restart="checkpoint.h5")`
- **`.dat` files** - Time series data (lift, drag, energy)
  - Plot with: `np.loadtxt("stats.dat")`

## Documentation

Each environment has detailed README.md:

- [cavity/README.md](cavity/README.md) - Lid-driven cavity
- [cylinder/README.md](cylinder/README.md) - Circular cylinder
- [pinball/README.md](pinball/README.md) - Three cylinders
- [step/README.md](step/README.md) - Backward-facing step

These include:
- Physical description
- Expected results
- Validation data
- Usage tips
- References

## Common Parameters

Most examples accept similar parameters (check each file for specifics):

**Reynolds number:**
- Cavity: 1000-7500 (turbulent)
- Cylinder: 40-200 (vortex shedding)
- Pinball: 30-100 (complex wake)
- Step: 400-800 (separated flow)

**Mesh resolution:**
- `coarse` - Fast, less accurate (~5k elements)
- `medium` - Good balance (~20k elements) **[default]**
- `fine` - Accurate, slow (~80k elements)

**Time step:**
- Typical: dt = 0.01
- High Re: dt = 0.001-0.005
- Rule: Keep CFL < 1

## Troubleshooting

**Import errors:**
```bash
# Ensure Firedrake environment is activated
source firedrake/bin/activate
pip install -e /path/to/hydrogym
```

**Compilation errors:**
```bash
# If using Docker/EasyBuild, load required modules first
module load GCC OpenMPI PETSc
```

**Simulation diverges:**
- Reduce time step: dt = 0.005
- Use coarser mesh: `mesh="coarse"`
- Check CFL number (should be < 1)

**Out of memory:**
- Use coarser mesh
- Reduce number of MPI processes
- Close other applications

## Contributing

To add a new example:

1. Follow the standard patterns (see existing examples)
2. Add clear docstring explaining purpose
3. Include in environment README.md
4. Test with `--dry-run` or small parameters first

## References

**Hydrogym:**
- Paper: [Add citation when published]
- GitHub: https://github.com/hydrogym/hydrogym

**Firedrake:**
- Website: https://www.firedrakeproject.org/
- Paper: Rathgeber et al. (2016)

**Flow Control:**
- Cavity: Ghia et al. (1982)
- Cylinder: Rabault et al. (2019)
- Pinball: Deng et al. (2020)
- Step: Armaly et al. (1983)

See individual environment READMEs for detailed references.

## License

See LICENSE file in repository root.

---

**Ready to get started?**

1. Test installation: `cd env_test && python test_firedrake_env.py --environment cylinder --num-steps 10`
2. Run first example: `cd cylinder && python run-transient.py`
3. Explore other environments: Check individual README.md files
4. Learn more: Read the [main documentation](../../../docs/)
