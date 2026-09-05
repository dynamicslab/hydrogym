---
sidebar_label: flow
title: hydrogym.firedrake.flow
---

## ScaledDirichletBC Objects

```python
class ScaledDirichletBC(fd.DirichletBC)
```

Dirichlet boundary condition whose value can be rescaled in place.

Wraps a Firedrake ``DirichletBC`` and multiplies the prescribed value
``g`` by an internal ``fd.Constant`` scale factor, so that the amplitude
of a boundary condition (e.g. actuation blowing/suction) can be changed
during a simulation without rebuilding the boundary condition. Call
``set_scale`` to change the factor.

**Attributes**:

- `unscaled_function_arg` - The original (unscaled) value ``g`` passed in.

#### \_\_init\_\_

```python
def __init__(V, g, sub_domain, method=None)
```

Construct the scaled boundary condition.

**Arguments**:

- `V` _fd.FunctionSpace_ - Function space the condition applies to.
- `g` - Value to prescribe; may be a UFL expression. The stored
  condition is ``self._scale * g``.
- `sub_domain` - Mesh marker(s) of the boundary facets.
- `method` - Boundary condition method, for compatibility with older
  Firedrake versions whose ``DirichletBC`` still accepts it.
  Newer versions ignore it.
  

**Notes**:

  Firedrake&#x27;s constructor signature changed across versions, so
  several calling conventions are tried in turn.

#### set\_scale

```python
def set_scale(c)
```

Set the multiplicative factor applied to the prescribed value.

**Arguments**:

- `c` - New scale factor; assigned to the internal ``fd.Constant``
  so it can be updated without rebuilding the condition.

## ObservationFunction Objects

```python
class ObservationFunction(NamedTuple)
```

Observation function paired with the number of values it produces.

**Attributes**:

- `func` _Callable_ - Function mapping a flow state to a numpy array.
- `num_outputs` _int_ - Size of the array returned by ``func``.

#### \_\_call\_\_

```python
def __call__(q: fd.Function) -> np.ndarray
```

Evaluate the wrapped observation function on a flow state.

**Arguments**:

- `q` _fd.Function_ - Flow state to observe.
  

**Returns**:

- `np.ndarray` - Observation values computed by ``func``.

## FlowConfig Objects

```python
class FlowConfig(PDEBase)
```

Base class for incompressible Navier-Stokes flow configurations.

A ``FlowConfig`` owns the mesh, the mixed velocity/pressure function
space, the current state ``q``, the boundary conditions, and the
observation function used for RL feedback. Concrete environments
(cavity, cylinder, pinball, step) subclass this and provide at minimum
``init_bcs``/``collect_bcu``/``collect_bcp``, ``linearize_bcs``, and
``configure_observations``.

Checkpoints can be resolved automatically: if no ``restart`` is given,
an environment name is inferred from the class name, Reynolds number,
and mesh, and a matching checkpoint is fetched from the Hugging Face
Hub via the data manager (falling back to a zero initial condition if
none is found). If the resolved environment has more than one
candidate checkpoint file, the pick is deterministic (sorted by
filename, last wins) but not guaranteed physically appropriate for
every use case -- pass an explicit checkpoint file path in ``restart``
if a specific one is required.

#### DEFAULT\_VELOCITY\_ORDER

Taylor-Hood elements

#### FUNCTIONS

tuple of functions necessary for the flow

#### \_\_init\_\_

```python
def __init__(velocity_order=None, **config)
```

Initialize the flow configuration.

**Arguments**:

- `velocity_order` _int, optional_ - Polynomial order of the
  Taylor-Hood velocity elements. Defaults to
  ``DEFAULT_VELOCITY_ORDER``.
- `**config` - Remaining configuration, consumed as keyword
  arguments:
  - Re (float): Reynolds number. Defaults to
  ``DEFAULT_REYNOLDS``.
  - probes (list): (x, y) probe locations used by the probe
  observation types; validated against the mesh on
  initialization unless ``validate_probes`` is False.
  - validate_probes (bool): Whether to check that ``probes``
  lie inside the mesh. Default True.
  - observation_type (str): Key selecting the observation
  function, forwarded to ``configure_observations``.
  - restart: Checkpoint to load: an explicit path, a
  Hugging Face Hub environment name, or a list of either.
  If None, a checkpoint is auto-inferred from the class
  name, Re, and mesh (and may be None if none is found).
  - mesh (str): Mesh name (e.g. &quot;medium&quot;). Defaults to
  ``DEFAULT_MESH``.
  - cache_dir, local_dir (str, optional): Custom cache and
  local fallback directories for checkpoint downloads.
  - use_HF_data_manager (bool): Whether to resolve
  checkpoints via the Hugging Face Hub. Default True.
  - hf_token (str, optional): Access token for private/gated
  Hub repos.
  - hf_revision (str, optional): Hub revision to pin
  downloads to.

#### load\_mesh

```python
def load_mesh(name: str) -> ufl.Mesh
```

Load a Gmsh mesh by name from ``MESH_DIR``.

**Arguments**:

- `name` _str_ - Mesh name, e.g. &quot;medium&quot;; reads
  ``{MESH_DIR}/`{name}`.msh``.
  

**Returns**:

- `ufl.Mesh` - The loaded Firedrake mesh.

#### save\_checkpoint

```python
def save_checkpoint(filename: str, write_mesh=True, idx=None)
```

Save the current state and actuator states to a Firedrake checkpoint.

Every function named in ``FUNCTIONS`` is written, along with the
state of each actuator as a file-level attribute &quot;act_state&quot;.

**Arguments**:

- `filename` _str_ - Path of the checkpoint file to write.
- `write_mesh` _bool, optional_ - Whether to also save the mesh.
  Default True.
- `idx` _int, optional_ - Time index to attach to the saved
  functions, if any.

#### load\_checkpoint

```python
def load_checkpoint(filename: str, idx=None, read_mesh=True)
```

Load the flow state (and actuator states) from a checkpoint.

Each function named in ``FUNCTIONS`` is read back onto the current
function space; if the checkpoint was written on a different
element, the field is projected onto the current space instead of
being assigned directly. Missing functions default to zero with a
warning. If a stored &quot;act_state&quot; attribute is present, it is used
to restore each actuator&#x27;s state.

**Arguments**:

- `filename` _str_ - Path of the checkpoint file to read.
- `idx` _int, optional_ - Time index of the saved functions to load.
- `read_mesh` _bool, optional_ - Whether to load the mesh from the
  file and reinitialize state. Default True; if False, the
  current mesh must already be initialized.

#### configure\_observations

```python
def configure_observations(obs_type=None,
                           probe_obs_types=`{}`) -> ObservationFunction
```

Select the observation function for this flow.

Subclasses should build a mapping of available observation types
(including the probe-based ones passed in ``probe_obs_types``) and
return the one named by ``obs_type``, raising ``ValueError`` for an
unknown type.

**Arguments**:

- `obs_type` _str, optional_ - Name of the desired observation type.
  Defaults to a subclass-specific choice.
- `probe_obs_types` _dict, optional_ - Mapping of probe-based
  observation type names to ``ObservationFunction`` objects,
  as constructed by ``FlowConfig.__init__``.
  

**Returns**:

- ``2 - The selected observation function.
  

**Raises**:

- ``3 - In the base class; subclasses must
  override this.

#### get\_observations

```python
def get_observations() -> np.ndarray
```

Compute the current observation vector.

**Returns**:

- `np.ndarray` - Output of the configured observation function
  evaluated on the current state.

#### num\_outputs

```python
@property
def num_outputs() -> int
```

Number of values in the observation vector.

This may be lift/drag, a stress &quot;sensor&quot;, or a set of probe
locations, depending on the configured observation type.

#### initialize\_state

```python
def initialize_state()
```

Create the function spaces and state functions on the current mesh.

Sets up the Taylor-Hood mixed space (continuous vector-valued
velocity of order ``velocity_order`` and continuous piecewise-linear
pressure), allocates one ``fd.Function`` per name in ``FUNCTIONS``,
breaks out the velocity and pressure subfunctions, allocates the
internal vorticity field, and validates any configured probe
locations.

#### set\_state

```python
def set_state(q: fd.Function)
```

Set the current state fields

**Arguments**:

- `q` _fd.Function_ - State to be assigned

#### copy\_state

```python
def copy_state(deepcopy: bool = True) -> fd.Function
```

Return a copy of the current state fields

**Returns**:

- `q` _fd.Function_ - copy of the flow state

#### create\_actuator

```python
def create_actuator(tau=None) -> ActuatorBase
```

Create a single actuator for this flow

#### reset\_controls

```python
def reset_controls(function_spaces=None)
```

Reset the controls to a zero state

Note that this is broken out from `reset` because
the two are not necessarily called together (e.g.
for linearization or deriving the control vector)

TODO: Allow for different kinds of actuators

#### nu

```python
@property
def nu()
```

Kinematic viscosity ``1 / Re`` as a Firedrake Constant.

#### split\_solution

```python
def split_solution()
```

Break the mixed state out into velocity and pressure fields.

Assigns ``self.u`` and ``self.p`` to the subfunctions of ``self.q``
and renames them to &quot;u&quot; and &quot;p&quot;, so that fields saved under those
names stay consistent after loading or reinitializing state.

#### vorticity

```python
def vorticity(u: fd.Function = None) -> fd.Function
```

Compute the vorticity field `curl(u)` of the flow

**Arguments**:

  u (fd.Function, optional):
  If given, compute the vorticity of this velocity
  field rather than the current state.
  

**Returns**:

- `fd.Function` - vorticity field

#### function\_spaces

```python
def function_spaces(mixed: bool = True)
```

Function spaces for velocity and pressure

**Arguments**:

  mixed (bool, optional):
  If True (default), return subspaces of the mixed velocity/pressure
  space. Otherwise return the segregated velocity and pressure spaces.
  

**Returns**:

  Tuple[fd.FunctionSpace, fd.FunctionSpace]: Velocity and pressure spaces

#### collect\_bcu

```python
def collect_bcu() -> Iterable[fd.DirichletBC]
```

List of velocity boundary conditions

#### collect\_bcp

```python
def collect_bcp() -> Iterable[fd.DirichletBC]
```

List of pressure boundary conditions

#### collect\_bcs

```python
def collect_bcs() -> Iterable[fd.DirichletBC]
```

List of all boundary conditions

#### epsilon

```python
def epsilon(u) -> ufl.Form
```

Symmetric gradient (strain) tensor

#### sigma

```python
def sigma(u, p) -> ufl.Form
```

Newtonian stress tensor

#### residual

```python
def residual(q, q_test=None)
```

Nonlinear residual for the incompressible Navier-Stokes equations.

Returns a UFL form F(u, p, v, s) = 0, where (u, p) is the trial function
and (v, s) is the test function.  This residual is also the right-hand side
of the unsteady equations.

A linearized form can be constructed by calling:
```
F = flow.residual((uB, pB), (v, s))
J = fd.derivative(F, qB, q_trial)
```

#### max\_cfl

```python
@pyadjoint.no_annotations
def max_cfl(dt) -> float
```

Estimate of maximum CFL number

#### body\_force

```python
@property
def body_force()
```

Volumetric body force term (zero by default).

**Returns**:

- `fd.Function` - A vector function on the velocity space, assigned
  to zero. Subclasses may override this to return a nontrivial
  forcing field.

#### linearize\_bcs

```python
def linearize_bcs()
```

Sets the boundary conditions appropriately for linearized flow

#### set\_control

```python
def set_control(act: ArrayLike = None)
```

Directly sets the control state

Note that for time-varying controls it will be better to adjust the controls
in the timestepper, e.g. with `solver.step(iter, control=c)`.  This could be used
to change control for a steady-state solve, for instance, and is also used
internally to compute the control matrix

#### inner\_product

```python
def inner_product(q1: fd.Function,
                  q2: fd.Function,
                  assemble=True,
                  augmented=False)
```

Energy inner product for the Navier-Stokes equations.

`augmented` is used to specify whether the function space is
extended to represent complex numbers. In this case the inner
product is the L2 norm of the real and imaginary parts.

#### velocity\_probe

```python
def velocity_probe(probes, q: fd.Function = None) -> list[float]
```

Probe velocity in the wake.

Returns a list of velocities at the probe locations, ordered as
(u1, u2, ..., uN, v1, v2, ..., vN) where N is the number of probes.

#### pressure\_probe

```python
def pressure_probe(probes, q: fd.Function = None) -> list[float]
```

Probe pressure around the cylinder

#### vorticity\_probe

```python
def vorticity_probe(probes, q: fd.Function = None) -> list[float]
```

Probe vorticity in the wake.

#### linearize

```python
def linearize(qB=None,
              adjoint=False,
              sigma=0.0,
              inverse=False,
              solver_parameters=None)
```

Return a linear operator for the Navier-Stokes equations about a base flow.

By default returns the Jacobian operator ``dF/dq`` evaluated at the
base state ``qB``. With ``inverse=True``, returns an operator that
solves the mass-matrix pencil ``(J - sigma * M) @ v1 = M @ v0``,
shifted by the (possibly complex) ``sigma``; complex shifts are
handled as a real block system with twice the degrees of freedom.

**Arguments**:

- ``0 _fd.Function, optional_ - Base state to linearize about.
  Defaults to the current state.
- ``1 _bool, optional_ - Whether to return the transpose
  (adjoint) of the operator. Default False.
- ``2 _complex, optional_ - Spectral shift. Default 0.0; must be
  zero unless ``inverse=True``.
- ``5 _bool, optional_ - Whether to return a shift-inverse
  (solve) operator instead of the Jacobian itself. Default
  False.
- ``6 _dict, optional_ - PETSc solver parameters for
  the inverse operators.
  

**Returns**:

  DirectOperator or InverseOperator: The requested linear
  operator (transposed if ``adjoint=True``).
  

**Raises**:

- ``9 - If a nonzero ``sigma`` is given with
  ``inverse=False``.

