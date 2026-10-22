# NekRS DRL channel: HydroGym control interface (2026-10-22)

The HydroGym Nek5000 coupling of TCFlarge_3D_Re207 (Nek5000 case `large_channel/drl/*.f` of the HydroGym image, spec in `../notes/channel_drl_spec.md`) ported to NekRS 26.0.
- `hydrogym/nek/env.py` (`NekEnv`) runs **unchanged**: verified by md5 against the copy in the HydroGym Nek5000 image.
- Every request (INTAL, STATE, CNTRL, EVOLV, RSETS, TERMN) has the same tags, counts, datatypes and timing as the Fortran.

## Files

| File | Role |
|---|---|
| `tcf.udf` | Port of `DRL_main` and the init, state, action, reward and IO routines. |
| `tcf.par` | Same numerics as `uncontrolled/tcf.par`, but the wall (bid 1) is `udfDirichlet`. CASEDATA holds ndrl, znmf, yPlusSensing, reTau, gllUnique, totctrl and statInterval. |
| `tcf_rcb.par` | Same, with `[MESH] partitioner = rcb`. The default partitioner puts the whole wall on one rank; `rcb` spreads it over ranks, which tests the multi-rank protocol. |
| `drl_client.py` | Test controller driving `NekEnv`. Policies: zero, opposition (`-alpha·v'`) and random. |
| `compare_runs.py` | Compares runs with the uncontrolled reference `uncontrolled/wall.dat`. |
| `uncontrolled/` | Uncontrolled NekRS channel (`tcf.par`, `tcf.udf` logging the plane-averaged wall shear) and its reference `wall.dat`. Run: `../podman.sh 'cd uncontrolled && mpirun -n 1 nekrs --setup tcf.par --backend CUDA'`. |
| `../mpmd/` | Patched NekRS driver `nekrs-mpmd`. With `NEKRS_MPMD_COLOR=1` it joins HydroGym's `COMM_WORLD.Split`, where the stock driver would deadlock. |
| `run.sh` | MPMD launcher, for example `GPU=3 NP=1 RUN=runs/x ./run.sh --policy zero --steps 20`. Use one GPU per NekRS rank. `PAR=tcf_rcb.par` selects the alternative par. |

## Port details

- **Registration:** every GLL node of every bid-1 face, in `register_wall_pts` loop order (iz, iy, ix), then a stable sort by x.
  - `gllUnique=1` gives the `facindf` subset.
  - The info fields (element, face, ix, iy, iz) use Nek5000 conventions: 1-based, preprocessor face numbering.
- **State:** u − ⟨u⟩ and v − ⟨v⟩, where ⟨·⟩ is the unassembled-mass-weighted x–z mean at each GLL y-level (the same as Nek5000 `planar_avg` z then x). They are interpolated with findpts at wall point + (0, 15/207, 0).
- **Action:**
  - receive the actions;
  - ZNMF: `dssum`, which adds duplicated copies, then subtract the wall mean;
  - uy = result on all wall faces, with ux = uz = 0.
- **Reward:**
  - element-local JW·∂u/∂y (`strongGrad` without averaging, which is `gradm1`);
  - wall plane mean;
  - the Fortran running mean (sample 1 double-weighted).
- **CFL:** `|COURNO|` uses the `cflHex3D` definition, computed on the host because `nrs_t::computeCFL` is private. It agrees with the NekRS step log.
- **RSETS** is a no-op that lets the solver advance one step, exactly as in Fortran.
- **TERMN** calls `MPI_Finalize` then `_Exit`.

## Verification (1 A6000 per rank; 13 solver steps per action ≈ 0.22 s)

| Test | Result |
|---|---|
| Rank count: 1 vs 2 vs 4 ranks (default partitioner) vs 4 ranks (rcb, wall split 1472/1408/1408/1472) | Rewards and observation statistics identical to float precision. |
| Zero action, 1500 actions (19,500 steps) vs uncontrolled `uncontrolled/tcf.par` | Wall dU/dy bit-identical to step 2080. Then chaotic divergence from round-off (1e-9 at step 3120, 2% at step 19,500). Window mean −0.29%. |
| ZNMF, random actions | Net wall flux 1e-16 on 1 and 4 ranks. |
| Opposition control (α = 1, sensing at y⁺ = 15), 1500 actions | Last-third dU/dy 10.02. That is **−21.9%** vs uncontrolled in the same window (−23.5% vs the YAML 13.10), in line with the ≈25% of Choi, Moin & Kim (1994). |

**Edge amplification measured in the solver (gllUnique=0, the released default):** commanded |a| ≤ 0.063 produces up to |uy| = 0.22 at the wall, because dssum adds the copies of edge and corner nodes.

The message backlog behaviour of the original code is reproduced too. Python never receives the CFL sent on STATE, so UCX reports one unmatched message per STATE request at shutdown.

## Not yet done / caveats

- **Actuator order differs from Nek5000.** It is per-rank, then sorted by x, and NekRS partitions differently from Nek5000's `.ma2`. Before evaluating a policy trained on Nek5000, check how the wrapper maps actuators: by index, or by `ACTUATOR_INFO` positions.
- **Pressure tolerance** is 1e-6, as in the validated uncontrolled run. The YAML has 1e-8, and the residual definitions differ between Nek5000 and NekRS anyway.
- **`znmf=0`:** at shared nodes NekRS combines Dirichlet copies with max/min sweeps, where Nek5000 uses per-copy values. This only matters for `znmf=0` with discontinuous actions, which is not the HydroGym config.
