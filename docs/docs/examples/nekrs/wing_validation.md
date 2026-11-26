---
sidebar_position: 1
---

# NekRS NACA0012 validation

The NekRS wing case (`examples/nekrs/wing`) is a port of the Nek5000 case `naca0012_200k`: NACA0012 at
Re_c = 2·10⁵, zero incidence, span 0.1 c with periodic sides. It uses Ref1's mesh at polynomial order 7. The boundary
conditions, trip and relaxation-term filter are ported line by line from the Nek5000 user routines. The uncontrolled
flow is validated against Ref1, the wall-resolved Nek5000 simulation of Wang, Atzori & Vinuesa (J. Fluid Mech.
2025), using the statistics of its KTH benchmark data.

**Source of the numbers.** The values on this page come from two uncontrolled NekRS runs, one with Ref1's trip (the
trip of `naca_wing.wall` and of `examples/nekrs/wing`) and one with a trip of 60 spanwise modes renewed every 0.004. Their statistics were computed with
`examples/nekrs/wing/bl_stats.py` and `bl_report.py`; the last column names the group of quantities in that output.

## Set-up

| | Ref1 | NekRS, Ref1's trip | NekRS, 60-mode trip |
|---|---|---|---|
| code | Nek5000 | NekRS 26.0 (GPU) | NekRS 26.0 (GPU) |
| mesh, N / dealiasing | Ref1, 7 / 12 | Ref1 (same file), 7 / 12 | Ref1, 7 / 12 |
| time stepper, Δt U∞/c | BDF3/EXT3, 2·10⁻⁵ | BDF3/EXT3, 2·10⁻⁵ | BDF3/EXT3, 8·10⁻⁵ |
| stabilisation | RT filter, 2 modes, −2000 | same (`hpfrt`) | same |
| initial field | – | Ref1 restart, t = 28 | Ref1 restart, t = 28 |
| trip: modes (span 0.1 c) / renewal (c/U∞) | 30 / 0.008 | 30 / 0.008 | 60 / 0.004 |
| trip: amplitude / x/c / radii (c) | 10 / 0.1 / 0.004, 0.001 | same | same |
| statistics window (c/U∞) | – | 17 (t = 31–48) | 19 (t = 43.5–62.5) |

## Results (NekRS with Ref1's trip against Ref1)

| Quantity | Agreement | Group |
|---|---|---|
| Suction-side C_f, nine stations x/c = 0.12–0.9 | within 3.4 % (largest at x/c = 0.40, in transition, where the 95 % interval of the NekRS mean is ±3.2 %); within 0.8 % from x/c = 0.5 on | Stations |
| C_p at the stations | within 3·10⁻⁴ (below 1.2·10⁻³ over x/c = 0.05–0.95) | Stations |
| Transition: C_f minimum / midpoint | x/c = 0.32 / 0.44 (Ref1 0.30 / 0.45) | Transition |
| H₁₂ and Re_θ at x/c = 0.3, 0.5, 0.7, 0.9 | within 0.8 % and 1.4 % | Boundary layer |
| Peaks of u_rms⁺, v_rms⁺, w_rms⁺, −u'v'⁺, same stations | within 7 % (within 3 % from x/c = 0.7 on, except −u'v'⁺ at 0.9: 5 %) | Profiles |
| Whole-wing drag | C_D = 0.01403 ± 0.00006 (C_D,f = 0.01123, C_D,p = 0.00280) against 0.0140 (0.0112, 0.0028) for Ref1 over the full surface (+0.2 %) | Forces |
| Suction-side strip mean C_f,e, x/c = 0.25–0.858 | 4.14·10⁻³ against 4.18·10⁻³ for Ref1 (−0.9 %) | Strip |
| Resolution at x/c = 0.3, 0.5, 0.7, 0.9 | first GLL point Δn⁺ ≤ 0.62, mean Δs⁺ ≤ 17.4, Δz⁺ ≤ 6.9, Δt⁺ ≈ 0.01 | Resolution |

The stored 80-station quadrature of Ref1's statistics file stops at x/c = 0.995 and misses part of the pressure
drag near the leading and trailing edges. It gives C_D = 0.01184, the value quoted in Ref1. The full-surface
values are the ones to compare with.

The defaults of `examples/nekrs/wing` are Ref1's trip and start field with Δt = 4·10⁻⁵ and 31 steps per action, the
deployment of the Nek5000 case with the observations in the wall-tangent frame (`wing.yml`); opposition control runs at Δt = 2·10⁻⁵ (`wing_oc.yml`).

## Sensitivity to the trip

The restart files of Ref1 carry a trip of 30 spanwise modes renewed every 0.008 c/U∞, as `naca_wing.wall`. With 60
modes renewed every 0.004 NekRS transitions later: the C_f minimum moves
from x/c = 0.32 to 0.37, the transition midpoint from 0.44 to 0.52, and H₁₂ < 1.7 is reached at x/c = 0.69 instead
of 0.57. The whole-wing C_D is 6.4 % lower (0.01313 ± 0.00010) and C_D,f 7.6 % lower. The trip therefore changes
the flow much more than the choice between the two codes does. `examples/nekrs/wing` uses Ref1's trip by default.
The 60-mode trip is an option (`TRIP_MODES=60 TRIP_INTERVAL=0.004`); results with it should not be compared with
Ref1.
