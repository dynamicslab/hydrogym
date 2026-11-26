---
sidebar_position: 1
---

# Actuation power

The controlled environments report the power their actuators put into the flow, in the units of their drag (or
objective) coefficient, and can charge that power in the reward. The default `power_weight = 0` leaves the
reward unchanged, so published configurations and results stay reproducible. Exception: the Nek5000 backend
(`hydrogym.nek`) reports the power for the wing cases only and only when its configuration has a `power` section
(the Hugging Face configurations of both wings have one; see [Environments](#environments)).

The shared definitions live in `hydrogym.power` (`power_forms`, `transpiration_forms`, `PowerConfig`,
`power_info`, `net_saving`).

## Definitions

Power put into the fluid by actuator *i* (positive = energy into the flow):

| Actuation | Power |
|---|---|
| rotating body | $P_i = \Omega_i M_i$, the shaft power; $M_i$ is the moment the body exerts on the fluid (the motor torque) |
| wall transpiration / jets, normal velocity $v_n$ into the fluid | $P = \int_A v_n \left(p - p_\mathrm{ref} + \tfrac12 \rho \lvert u\rvert^2\right) \mathrm{d}A$ |
| volume forcing $f$ | $P = \int_V f \cdot u \,\mathrm{d}V$ |

For jets, $p_\mathrm{ref}$ is the instantaneous mean wall pressure over the group of jets that together have
zero net mass flux (e.g. the jet pair of a cylinder). With zero net mass flux the signed total does not depend on
$p_\mathrm{ref}$; with $p_\mathrm{ref}$ equal to the group mean, the clipped forms below do not depend on the
pressure gauge either.

### Cost forms

| form | rigid bodies (per body) | jets (per surface point, $\pi_p = v_n (p - p_\mathrm{ref})$, $\pi_k = v_n \lvert u_w\rvert^2/2$) |
|---|---|---|
| `signed` | $\sum_i P_i$ | $\sum w\,(\pi_p + \pi_k)$ |
| `positive` | $\sum_i \max(P_i, 0)$ | $\sum w\,(\max(\pi_p,0) + \max(\pi_k,0))$ |
| `abs` | $\sum_i \lvert P_i\rvert$ | $\sum w\,(\lvert\pi_p\rvert + \lvert\pi_k\rvert)$ |

`signed` credits energy recovered by an actuator, `positive` follows Hasegawa & Kasagi (2011, App. A), `abs` is the
conservative estimate of Bewley, Moin & Temam (2001). The forms are built for every time sample and then
averaged in time.

### Normalisation

$$C_P = \frac{P}{\tfrac12 \rho U_\mathrm{ref}^3 L_\mathrm{ref}\,[\times\,\mathrm{span}]}$$

i.e. the factor that turns the environment's drag force into $C_D$. The drag power coefficient is then $C_D$
itself and the net saving against an uncontrolled reference is

$$S = 1 - \frac{C_D + C_P}{C_{D,0}} = \mathrm{DR} - \frac{C_P}{C_{D,0}}$$

(`hydrogym.power.net_saving`).

## Configuration

Keys in the environment configuration (Firedrake: the `env_config` dict of `FlowEnv`; JAX: `env_config`;
JAX-Fluids: the `jaxfluids` section of `environment_config.yaml`, overridable in `env_config`):

| key | default | effect |
|---|---|---|
| `power_weight` | `0.0` | reward ← reward − `power_weight` · C_P[`power_cost`] |
| `power_cost` | `"abs"` | `signed`, `positive` or `abs` |
| `power_sampling` | `"mean"` (m-AIA: `"end"`) | `"mean"`: mean over the solver steps of the action interval; `"end"`: instantaneous at its end. m-AIA samples only at the end of the action and raises an error for `"mean"` |

## `info` keys

`step()` (and `reset()` where the API has an `info` dict) return

| key | meaning |
|---|---|
| `actuation_power` | C_P, `signed` |
| `actuation_power_positive` | C_P, `positive` |
| `actuation_power_abs` | C_P, `abs` |
| `actuation_power_per_actuator` | signed C_P per actuator |
| `drag_coefficient` | C_D of the controlled body/bodies (only where the environment has a drag) |
| `power_sampling` | `"mean"` or `"end"` |
| `objective` | the environment's objective quantities |

The JAX environments follow the gymnax API: `step` is jit-compiled, so the values are arrays and
`power_sampling` is a static string (`hydrogym.jax.utils.power.StaticStr`); `reset` has no `info`.

## Environments

| backend / env | actuators | power | $U_\mathrm{ref}$, $L_\mathrm{ref}$ | drag |
|---|---|---|---|---|
| Firedrake `RotaryCylinder` | 1 rotating cylinder | $\Omega M$ | $U_\infty$, $D$ | $C_D$ |
| Firedrake `Pinball` | 3 rotating cylinders | $\Omega_i M_i$ | $U_\infty$, $D$ | $\sum C_{D,i}$ |
| Firedrake `Cylinder` | 2 jets (upper, lower) | jets, $p_\mathrm{ref}$ = mean over both jets | $U_\infty$, $D$ | $C_D$ |
| Firedrake `Cavity` | 1 slot | jets, $p_\mathrm{ref}$ = mean slot pressure | $U_\infty$, cavity length | – |
| Firedrake `Step` | 1 slot | jets, $p_\mathrm{ref}$ = mean slot pressure | $U_\max$, step height | – |
| JAX `KolmogorovFlow` | 4 forcing modes | $\langle f_i\cdot u\rangle$ (per unit area) | 1, 1 | – |
| JAX `ChannelFlowSpectralEnv` | 24 wall jets | $w_\mathrm{jet} (p - p_\mathrm{ref} + w_\mathrm{jet}^2/2)$ on the wall, $p_\mathrm{ref}$ = wall mean | $U_b$, $L_x L_y$ (area) | $C_f$ |
| JAX-Fluids `Nozzle2D/3D` | 2 / N injectors | exergy flow $\dot m R T_\infty \ln(p_t/p_\infty)$ | ideal primary-jet exergy flow | – |
| m-AIA `Pinball`, `RotaryCylinder` (2D, 3D about z) | rotating cylinders | $\Omega_i M_i$, torque from the angular-momentum balance on circles at 1.3–1.6 radii (cell-centre values) | $U = Ma/\sqrt3$, $D$ (× span) | $\sum C_{D,i}$ |
| m-AIA `Cylinder`, `JetPinball` | slot jets on cylinders | jets, wall pressure extrapolated from 1.5–4.5 cells, $p_\mathrm{ref}$ = mean over the jets of a plane | $U$, $D$ (× span) | $C_D$ |
| m-AIA `Cavity`, `Cavity3Jet` | 1 / 3 slot jets (independent) | jets, the fluid cells next to the slot (BC 2007 cells), $p_\mathrm{ref}$ = mean over the jets | $U$, cavity depth $D$ | – |
| m-AIA `SquareCylinder` (2D, 3D), `Cube` | 2 / 4 slot jets on flat faces (independent) | jets, the fluid cells next to the faces | $U$, $D$ (× span) | $C_D$ |
| m-AIA `NACA0012`, `NACA0012Gust` (2D, 3D) | 3 jets (independent) | jets, link-level model of the moving-wall bounce-back incl. its interpolation terms (case geometry file; 3D: mid-span section extruded) | $U$, $c$ (× span) | $C_D$ |
| m-AIA `Sphere` | 10 ring jets (independent) | jets, link-level model on the analytic sphere, wall pressure from 2–5 cells | $U$, $\pi D^2/4$ | $C_D$ |
| m-AIA `DRA2303SurfaceWave`, `ZPGTBLSurfaceWave` | travelling surface wave | power coefficient returned by the solver | scaling of the env's $C_D$ | $C_D$ |
| m-AIA `DRA2303Jet`, `ZPGTBLJet` | jets of the structured finite-volume solver (bc1008, independent) | jets, per wall face in the solver, wall pressure extrapolated from 2 cells, $p_\mathrm{ref}$ = mean over the jets | scaling of the env's $C_D$ | $C_D$ |
| NekRS wing and channel | wall transpiration, per GLL node | jets, point-wise; $p_\mathrm{ref}$ = spanwise mean per station | $U_\infty$, chord × span (wing) | $C_D$ |
| Nek5000 wing cases (`hydrogym.nek`, `NACA0012_3D_Re200000_AOA0`, `NACA4412_3D_Re75000_AOA5`) | wall transpiration, per wall GLL node | jets, point-wise, as NekRS; $p_\mathrm{ref}$ = spanwise mean per station | `power.reference` (default: area of the wing faces) | – |
| Nek5000 channels (`hydrogym.nek`) | wall transpiration | not available | – | – |

Notes:

* Nek5000 wing cases: the power is computed in the solver (`drl/drl_power.f` of the case) from the imposed wall
  velocity and the wall pressure at every GLL node of the wing faces, and returned to `hydrogym.nek` on the request
  `POWER`. It is enabled by a `power` section in `environment_config.yaml` (keys of the table above and
  `reference: {velocity, length, span}` or `{velocity, area}`); the section sets `simulation.power_mode` (p096 of the
  `.rea` file: 1 every solver step, 2 end of the action), which the parameter writer of the case puts into the
  `.rea` file. The `.rea` files of both wing environments have p096 = 1, which returns both samplings. The executable must be built from a case that contains `drl/drl_power.f`; an older executable
  does not answer `POWER`, and `hydrogym.nek` then stops with an error after `power.reply_timeout` (default 600 s). In the PettingZoo wrapper
  every agent's `info` holds the totals and its own signed value (`actuation_power_agent`).
* The Firedrake `Cylinder` jets are a zero-net-mass-flux pair (a positive control blows through the upper jet and
  sucks through the lower one with the same flow rate), so the power is complete and gauge-independent. In
  HydroGym ≤ 1.0 both jets blew with the same flow rate; `flow_config={"legacy_jets": True}` reproduces that. The
  cavity and step actuators are, by design, a single slot (the leading-edge slot of Barbagallo et al. 2009, net flow
  $0.078\,U L$ per unit control): with these net-flow actuators (and the legacy cylinder jets) the group-mean
  $p_\mathrm{ref}$ makes the reported power gauge-independent, but it does not include the work $p_\mathrm{ref} Q$ of
  supplying the net flow $Q$.
* Kolmogorov: the power is computed from the vorticity source the solver adds, so it is always the power of the
  force actually applied. By default that force is exactly $a_i \sin(k_i y)\,\mathbf{e}_x$. The `flow_config`
  switches `legacy_control_scaling=True`, `legacy_grid_endpoint=True` and `legacy_dealiasing=True` reproduce the
  control scaling, grid and advection term of HydroGym ≤ 1.0 (see the Kolmogorov example).
* Channel: the jets are a zero-net-mass-flux transpiration velocity $w_\mathrm{jet}(x, y)$ on the bottom wall, imposed
  consistently in the pressure projection, with a gain that ramps up and down over 10 DNS steps at the start and end
  of every action; at zero action the environment is the uncontrolled channel. The power is the energy flux through
  the wall, formed per wall point (the viscous normal stress vanishes at a no-slip wall); the kinetic-energy budget
  closes to the discretisation error (4 % at $32^3$ right after the ramp, 1 % later). `env_config["legacy_jets"] = True`
  reproduces the jet model of HydroGym ≤ 1.0, which sets the wall-normal velocity in the five grid layers next to the
  bottom wall after every stage; its power is the rate of kinetic-energy change of that constraint,
  $\sum (w_\mathrm{jet}^2 - w_\mathrm{free}^2)/2\,\mathrm{d}V/\mathrm{d}t$.
* m-AIA jets (BC 2007, `hydrogym.maia.wall_jets`): the jet cells and their wall velocities follow the solver's
  rule for every object class (box, profile across the slot, direction). On walls on cell faces the jet is exactly
  the layer of fluid cells next to the wall. On curved walls (airfoil, sphere) and walls off the cell faces the
  model follows the bounce-back link by link: which boundary cell writes each cut link, its moving-wall source, and
  the flow-dependent mass flux of the interpolated bounce-back (evaluated from the probed cell values), which on the
  NACA 0012 at $c/128$ (jet centres inside the airfoil) is 12–36 % of the jet mass flux. The Cavity, SquareCylinder,
  Cube, NACA0012 and Sphere jets are driven independently (not zero-net-mass-flux); as for the Firedrake cavity and
  step slots the group-mean $p_\mathrm{ref}$ makes the power gauge-independent but leaves out $p_\mathrm{ref} Q$.
* m-AIA `DRA2303Jet`, `ZPGTBLJet`: the structured finite-volume solver forms the jet power itself, per wall face of
  the jet footprints (`MaiaInterface.getActuationPower`, `power_model = "solver_jets"`): the wall velocity bc1008
  prescribes at the end of the action (the action once the ramp is complete) times the face metric oriented into
  the fluid, the wall pressure and density extrapolated linearly from the first two cells (as for $c_p$), and
  $p_\mathrm{ref}$ the area mean of the wall pressure over all jets. The jets are driven independently, so, as for
  the cavity, the power leaves out $p_\mathrm{ref} Q$. The viscous work at the wall is not included. The $C_P$ of
  the solver's force vector (the moving-wall power used by the surface-wave cases) stays 0 for the jets, so the
  rewards are unchanged.
* Nozzle: compressible injection from a reservoir at $T_\infty$; the exergy flow (minimum supply work) reduces to the
  incompressible $Q(p_t - p_\infty)$ at low pressure ratio. The thrust-angle objective has no force coefficient, so
  $C_P$ is normalised by the exergy flow of the ideal choked primary jet.

## Accuracy

* All backends form the cost forms with the same helper (`hydrogym.power`); the m-AIA and NekRS transpiration models
  give identical results for identical wall data (to 1e-12, for any pressure gauge).
* NekRS (100 actions of the NACA0012 wing with opposition control): with power sampled at every solver step, the absolute
  form exceeds the end-of-action value by a median 3.8 (up to 60 in the first actions after a start): the pressure
  response to each jump of the wall velocity carries work in the first solver steps of every action.
* m-AIA probes are sampled by nearest neighbour; the power models therefore probe cell centres and interpolate
  themselves. Rotation, pinball Re 100: $C_P$ 3.132–3.138 for five annuli (0.2 %); with vortex shedding the time
  means agree within 0.4 % (a single sample also contains the change of the angular momentum of the annulus fluid).
  Jets, cylinder Re 1000 (slot of 13 cells): within 0.8 %; Re 200 (slot of 4 cells): ±10 % (exit pressure
  under-resolved). `info["actuation_power_model_spread"]` reports the self-check of every m-AIA sample.
* m-AIA jets on flat walls, airfoils and spheres (2D cases, constant actions; the environment's
  info equals the stand-alone model exactly). Jet mass flux of the model against a control-volume balance of the
  solver's flow: Cavity 1.8 %, Cavity3Jet 1.3–3.3 %, SquareCylinder Re 1000 1 % and Re 200 5 %, NACA0012 Re 100
  0.7–3.8 % (without the interpolation terms of the bounce-back 12–36 % too low). Power over seven wall stencils (0.5–8 cells):
  SquareCylinder Re 1000 (slots of 12.8 cells) ±1.2 %; NACA0012 absolute form ±0.8 % (signed total a residual
  below 5e-4 at $C_D$ 0.57); slots of 4–5 cells (SquareCylinder Re 200, cavities) are not resolved: the signed power
  changes by a factor 2–3 between stencils (`actuation_power_model_spread` 0.6–1). For the 3D cases (Cube, Sphere,
  SquareCylinder and NACA0012 3D) the jet geometry is checked against the solver's rule in the unit tests.
* Firedrake: shaft power from the wall traction against a control-volume momentum balance 0.26 %; jet kinetic flux
  against the closed form 1 %. Kolmogorov: energy budget closes to 1e-6.
* Cross-solver, saturated pinball at Re 100: m-AIA (90d × 80d, d/20) and Firedrake (60d × 60d, d/30) agree in drag
  within 0.4 % and in shaft power within 5 % (3.138 and 2.975).

**Limitation of the m-AIA rotation model.** The ring torque assumes that the angular momentum of the fluid between
the body and the circles is steady at the end of the action. That holds for steady and slowly varying rotation (all
checks above); when the rotation command changes by more than 10 % of the bound within one action (e.g. bang-bang
reversal), the sample also contains the change of that angular momentum and can even be negative. Such samples are
flagged with `info["actuation_power_valid"] = False` (and `info["actuation_power_command_change"]`), and charging
them in the reward raises a warning. The jet models, NekRS and Firedrake (wall traction) have no such restriction.

## References

* T. R. Bewley, P. Moin, R. Temam (2001), DNS-based predictive control of turbulence: an optimal benchmark for
  feedback algorithms, *J. Fluid Mech.* 447, 179–225.
* Y. Hasegawa, N. Kasagi (2011), Dissimilar control of momentum and heat transfer in a fully developed turbulent
  channel flow, *J. Fluid Mech.* 683, 57–93.
* M. Quadrio, P. Ricco (2004), Critical assessment of turbulent drag reduction through spanwise wall oscillations,
  *J. Fluid Mech.* 521, 251–271.
