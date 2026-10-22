> Written during the NekRS reproduction of the HydroGym Nek5000 cases. File references (`file:line`) are to
> the Nek5000 case `large_channel/` (HydroGym image, `Nek5000/1.0-gompi-2024a-SystemCUDA-LargeChannel`); paths such as `container_case_full/` and `channel_case/` are the local copies used at the time.

# DRL coupling spec of the HydroGym Nek5000 channel (TCFlarge_3D_Re207), extracted from the Fortran sources

Source: `channel_case/large_channel` (container copy of the case). Line references are to those files. Extracted 2026-10-22.

**Main point:** the Fortran side never scales, clips, interpolates or smooths anything.
- It holds each action for exactly ndrl = 13 steps.
- It sends raw u′, v′ and the raw plane-averaged wall dU/dy.
- It imposes the received values directly as the wall-normal velocity.

`rescale_actions`, `normalize_input`, `u_tau`, `dUdy`, `npl_state`, `nxs` and `nzs` exist only on the HydroGym Python side.

## 0. Active build
- **Preprocessor flags:** `makefile:18` has `PPS=MPIIO DRL TSRS`, matching the EasyBuild recipe (`PPLIST='MPIIO DRL TSRS'`).
  - **BDFD, CTRL, OPPO and YWDEBUG are undefined.**
  - `compile_script` (`MPIIO DRL BDFD TSRS`, links `bdforce.o`) was not the build used.
- **Compiled files:** `misc.o`, `drl_{main,state,reward,IO,action,init}.o`. `bdforce.o`, `oppo_ctrl.o` and `init_CURSOR.o` are not built.
- **uparam values the Fortran reads:**

  | uparam | Meaning |
  |---|---|
  | (1) | ndrl |
  | (2) | ZNMF flag |
  | (3) | y+ of the sensing plane |
  | (4) | Re_τ |
  | (5–7) | BDFD only (inactive) |
  | **(8)** | `gll_unique`: not in `tcf.par`, therefore **0** |

- **Hard-wired:**
  - `totctrl=128`; `xnel/ynel/znel = 10/8/9`; `lp=512`, `lelt=3`.
  - Constant bulk velocity: `param(54)=-1`, `param(55)=1`. The body force is zero.

## 1. Control-loop timing and MPI
- `DRL_main` is called from `userchk` every step.
- **ISTEP=0:** creates the intercommunicator (tag 99) and runs `drl_init`. There is no exchange.
- **Every step after that:**
  1. While evolving, it counts `i_evolv` up to ndrl.
  2. It then serves requests until it receives `EVOLV`. Rank 0 receives a 5-character request (tag 22) and broadcasts it.
- **Requests:**
  - `STATE`, `CNTRL`, `INTAL` and `EVOLV`;
  - `TERMN` (`exitt0`; prints "CKPT SAVED" but writes no checkpoint);
  - `RSETS` (a no-op, see §5).
- **Policy steps** are at ISTEP = 1, 14, 27, … New actions drive the boundary conditions of steps s+1 … s+13.
- **Actions are held constant** between policy steps. `moving_smooth_action` is commented out, so there is no ramp or interpolation.
- **Messages:**
  - **INTAL:**
    - `node_list(LP)` (tag 1996);
    - per rank: NUMCTRL and `info_agt(1..5)` = element, face, ix, iy, iz;
    - `pos_agt` x, y, z.
  - **STATE:** TIME (tag 1998), |COURNO| (tag 1999), then u′ and v′ per rank (tag `NID*il+70000`; on rank 0 both fields use tag 70000).
  - **CNTRL:** TOTCTRL doubles per rank (tag NID+90000).
  - **Reward:**
    - every evolve step, rank 0 sends CFL (tag 1999), so 13 of these per cycle;
    - at `i_evolv==ndrl`, `rwd_agt` (tag NID+80000).

## 2. Agents and actuation
- **Registration:** every GLL node of every `W` face is registered, and the face's BC is switched to `'v  '`.
  - **`gll_unique=0` (default):** 64 nodes/face (`facind`) with duplicated element-edge nodes, giving 90 faces × 64 = **5760 points**.
  - **`gll_unique=1`:** 49/face (`facindf`), giving **4410 unique nodes** (70×63).
  - Points are bubble-sorted by x within each rank.
- **The Fortran has no notion of agents.** It exchanges one value per GLL point; how points map to agents (`nxs`, `nzs`) is decided in Python.
- **ACTUATE_JET:** `uy = ACTIONS*msk_act` for y ≤ 0.1, with ux = uz = 0.
- **ZNMF (flag = 1):**
  - `ACTIONS = dssum(ACTIONS) − ⟨dssum(ACTIONS)⟩`, where ⟨·⟩ is the bm1-weighted planar average over the whole bottom wall;
  - the mask is 1 on all nodes of agent faces.
- **How ⟨·⟩ is computed** (`znmf_avg` in `drl_action.f`; Nek5000 `planar_avg`, navier5.f:3427; `gtpp_gs_setup`, dssum.f:788):
  - It is `planar_avg` with the z handle (`gtpp_gs_setup(igs_z, xnel*ynel, 1, znel, 3)`), then with the x handle (`(igs_x, xnel, ynel, znel, 1)`). Both flags are hard-wired `.TRUE.` in `inc_src/DRL`.
  - Each pass computes `ua = gs_op(bm1*u) / gs_op(bm1)`. The sum runs over every local copy of each node in the line, element-interface duplicates included, with the unassembled `bm1` as weight. That is exact GLL quadrature along the line.
  - The result is the GLL-quadrature x–z plane mean at each GLL y-level. This needs the lexicographic element numbering of the box mesh (10×8×9).
  - The input is `dssum(ACTIONS)`, which is continuous. So at the wall level the mean equals the area-weighted surface mean of `dssum(ACTIONS)` over the bottom wall, because all wall elements share one dy. The ∫ of the imposed velocity over the wall is then exactly zero, whatever the double counting.
- **With `gll_unique=0`, dssum adds the duplicated copies.**
  - Edge nodes get a+b, and corners get the sum of 4 neighbours.
  - This is a sum, not an average: the effective wall velocity on element edges is about 2× (corners about 4×) the commanded value.
- **There is no scaling or clipping in Fortran.** After the mean is removed, values can exceed ±0.063.

## 3. State
- **Fields:** u′ = u − ⟨u⟩ and v′ = v − ⟨v⟩.
  - ⟨·⟩ is the instantaneous bm1-weighted x–z plane average at each GLL level. There is no running mean.
- **Sensing height:** y = UPARAM(3)/UPARAM(4) = 15/207 = 0.07246, in the second element row.
- **Sampling:** one sensing point per registered wall node, evaluated spectrally with `findpts` (tol 5e-13).
- **No normalisation** in Fortran. The 2 fields per point are hard-wired (`NFLDC=2`).
- **`int_pos` is not used by the DRL code.**
  - Format: `#iv1 8 3 11520`, 80×72 points on y=0 and y=15/185.
  - It is the KTH-Toolbox TSRS point file, presumably for time-series output.

## 4. Reward
- **What is computed:** `gradm1(vx)` gives ∂u/∂y. It is then bm1-weighted plane-averaged at the wall.
  - Every agent gets the same global wall-mean dU/dy. It is raw.
- **Running mean over the cycle:** `rwd = (rwd*i + x_i)/(i+1)`, giving r = (2x₁ + x₂ + … + x₁₃)/14.
  - Sample 1 is double-weighted.
  - The sample window is steps s…s+12, while the action acts over s+1…s+13.
- **Python-only parts:** the reference dUdy = 13.10 (= 0.063²·3300) and the reward formula.

## 5. Reset
- `RSETS` sets a flag that is overwritten every call and never read. `drl_chkpt_reset` is never called, so the Fortran reset is a no-op.
- Real resets are TERMN plus a relaunch from the checkpoint (`READCHKPT=yes`, `CHKPFNUMBER=1`).
- Episode length (`nb_interactions: 2500`) is set in Python.

## 6. Opposition control (`oppo_ctrl.f`, not compiled)
- **How it works:**
  - It runs every step with no hold.
  - It senses v′ only, with uniform-weight plane means.
  - Sensing height is `UPARAM(1)/180` (hard-coded ν=2e-5, Re_τ=180).
  - The BC is uy = −UPARAM(2)·v′.
- **It could not have worked as configured:**
  - CTRL and OPPO are separate ifdefs;
  - it is not linked;
  - its subroutine names clash with `drl_init.f`;
  - the ACTION sentinel is uninitialised;
  - UPARAM(1)/(2) have different meanings there.
- **Intended scheme:** v′ at y+=15 → uy_wall = −α·v′, every step.

## 7. Body-force damping (`bdforce.f`)
Inactive: not defined, not linked, and UPARAM(5–7)=0.

## 8. Inconsistencies relevant to reproducibility
1. **ZNMF dssum double-counts edge and corner nodes** with `gll_unique=0`, the default unless HydroGym writes `userParam08`. The actuation amplitude seen by the flow then depends on the GLL grid, not only on the policy output.
2. **Δt+ per DRL step is 0.68**, not the 0.6 stated in the YAML and the paper.
3. **The reward running mean is mis-weighted** (sample 1 counts twice) and offset by one step from the action window.
4. **Sensing at 15/207 vs `int_pos` at 15/185.** `int_pos` is unused by DRL. The YAML has `Ny: 9`, but the mesh has 8 y-elements.
5. **Leftovers in `useric`** (harmless when restarting from a checkpoint): `Re_tau=173`, ν=2e-5, `Ubar=0.135`, `dy` always ≤ 0.
