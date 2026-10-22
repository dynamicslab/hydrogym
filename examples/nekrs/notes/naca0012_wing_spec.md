> Written during the NekRS reproduction of the HydroGym Nek5000 cases. File references (`file:line`) are to
> the Nek5000 case `naca0012_200k/` (HydroGym image, `Nek5000/1.0-gompi-2024a-SystemCUDA-NACA0012`) and the Hugging Face environment `NACA0012_3D_Re200000_AOA0`; paths such as `container_case_full/` and `channel_case/` are the local copies used at the time.

# NACA0012 wing (Re_c = 200k), Nek5000 LES case to NekRS port: specification

## Scope and conventions

**Sources (all read-only).** Paths are relative to `naca_wing_case/`.
- `C/` stands for `container_case_full/`. This is the full case that was compiled in the container.
- `CC/` stands for `container_case/`. It holds the mesh `naca_wing.re2` and `stat_pts.in`.
- `HF/` stands for `hf_case/NACA0012_3D_Re200000_AOA0/`. Its `naca_wing.rea`, `forparam.i` and `naca_wing.wall` are byte-identical to the copies in `C/`.
- `README.md`.
- `../channel_case/DRL_SPEC.md`.

**Conventions.**
- `file:line` references point into these sources.
- **FLAG** marks a suspected bug or inconsistency. None of them has been fixed.
- **UNKNOWN** marks a fact that the allowed sources cannot establish.
- Scripts used: `/tmp/re2_bc.py` and `/tmp/wall_agents.py` (re2 parsing in pure Python).

---

## 0. Build actually used

**Build flags.**
- `C/compiler.out`: compiled as `naca0012_200k` with `-DMPIIO -DDRL -DUTAU`, real*8.
- `PPLIST` is set at `C/compile_script:71`.
- **YWDEBUG is undefined**, so every `#ifdef YWDEBUG` block is inactive.

**Objects linked.**
- `IO_tools`, `chkpoint`, `uparam`, `statistics_2D` (+`IO`, `usr`, `debug`), `time_series` (+`IO`), `pts_redistribute`, `err_est`, `rt_filter`, `tripf`, `misc`, `drl_{main,action,init,reward,state,IO}`.
- `oppo_ctrl` is **not** linked: `C/compile_script:110` is commented out.

**FLAG: size mismatch in COMMON /SPONGE/.** gfortran warns that the block has two sizes, 663552 vs 1327104 bytes.
- `userf` and `userchk` declare 6 arrays: `usponge, vsponge, wsponge, uss, vss, wss` (`C/naca_wing.usr:55-62, 335-343`).
- `userbc` and `useric` declare only 3 (`C/naca_wing.usr:187-189, 291-293`).
- This is harmless as long as the first three arrays line up, which they do.

`C/naca_wing.f` is the `.usr` file plus the `userqtl` routine that makenek adds.

**SIZE (`C/SIZE`).**

| Parameter | Value |
|---|---|
| `lx1` | 8, i.e. N=7 (`C/SIZE:18`) |
| `lxd` | 12 |
| `lx2` | `lx1-2`, i.e. PN-PN-2 |
| `ldim` | 3 |
| `lelg` | 220528 |
| `lp` | 4096 |
| `lelt` | 54 |
| `lhis` | 220 |
| `mxprev` / `lgmres` | 40 / 40 |
| `lorder` | 3 |
| DRL block | `totctrl=5000`; `xnel=10024`, `ynel=1`, `znel=22` (comment "For NACA0012 Wing 400k") |

**FLAG: minimum rank count.** With `lelt=54`, the case needs at least ceil(220528/54) = **4084 MPI ranks**. The config uses `nproc 4096`.

---

## 1. Parameters in effect

### 1.1 `.rea` (`C/naca_wing.rea`, identical to `HF/`)

The `.rea` is NEKTON 2.61, 3D, with 118 parameters. "Line" is the line number in the `.rea` file.

| p | Line | File value | In effect | Note |
|---|---|---|---|---|
| p001 | 5 | 1 | 1 | density |
| p002 | 6 | -4.0E+05 ("Re=400k") | **-2.0E+05** | HydroGym overwrites it at runtime (README; config `viscosity: -200000`). **FLAG:** the file value is wrong for this case. |
| p011 | 15 | 23 | UNKNOWN (set by HydroGym / episode) | NSTEPS. `stat_end` runs at ISTEP==NSTEPS (`stats/statistics_2D.f:109`). |
| p012 | 16 | -2.0E-05 | **-1.5E-05** | Fixed dt (negative). Overwritten by HydroGym (README; config `dt: -1.5e-5`). |
| p014/p015 | 18/19 | 0 / 500000 | UNKNOWN | IOSTEP. The config has `IOSTEP 5000`; whether HydroGym writes p015 is UNKNOWN. |
| p021 | 25 | 5e-6 | UNKNOWN | Pressure tolerance. The config has `p_residualTol 1e-6`; whether it is written into p021 is UNKNOWN. |
| p022 | 26 | 1e-8 | 1e-8 | Velocity tolerance (config `v_residualTol 1e-8`). |
| p026 | 30 | 0.4 | — | Irrelevant because dt is fixed. |
| p027 | 31 | 3 | 3 | BDF3/EXT3 |
| p066/p067 | 70/71 | 6/6 | | binary I/O |
| p069 | 73 | 10 | 10 | `UPRM_PRB`. Also reused as `istcount` (time-series buffer) at `stats/statistics_2D.f:1265`. |
| p070 | 74 | 3000 | 3000 | `L2FREQ`: lambda2 and `la2` outpost. The config `writeLA2 5000` may override it (UNKNOWN). |
| p071 | 75 | 0 | 0 | `FIXGEOM`, so `fix_mygll` is inactive. |
| p072 | 76 | 0 | 0 | `NEW_DT`, inactive. |
| p075 | 79 | 5000 | 5000 | `CHKPTSTEP` (config `CHKPINTERVAL 5000`). |
| p076 | 80 | 1 | 1 | `IFCHKPTRST` = restart on. |
| p078 | 82 | 2 | 2 | `rt_kut` |
| p079 | 83 | -2000 | -2000 | `rt_kai` |
| p080 | 84 | 1 | 1 | `rt_wght` |
| p081 | 85 | 0 | 0 | `rt_ifboyd = .false.` |
| p087 | 91 | 10 | 10 | `stat_comp` |
| p088 | 92 | 5000 | 5000 | `stat_outp` |
| **p089** | 93 | **0** ("MA: RESERVED?") | must equal ndrl=40 | `drl_step` (`drl/drl_main.f:37`, `drl/drl_IO.f:37`). **FLAG**, see §8. Whether HydroGym writes it is UNKNOWN. |
| **p090** | 94 | **10** ("YW: wrt_ctrl …") | 10, unless HydroGym writes `znmf_avg` | **FLAG:** the comment says time-series frequency, but the code uses `int(PARAM(90))>0` as the **ZNMF switch** (`drl/drl_action.f:38,123`). 10 means ZNMF on, which matches config `znmf_avg 1` anyway. |
| p091 | 95 | 5000 | | "outp_ctrl". No reader was found in the linked code (UNKNOWN). |
| p093 | 97 | 40 | | projection space |
| p094/p095 | 98/99 | 20/20 | | projection start |
| p099 | 103 | 4 | | dealiasing (new) |
| p101/p103 | 105/107 | 0/0 | | Nek explicit filter off. Stabilisation comes **only** from the RT term. |

**Logical switches.** IFFLOW T, IFHEAT F, IFTRAN T, IFNAV T, IFSTRS F, IFSPLIT F (PN-PN-2), IFCHAR F.

**Other `.rea` content.**
- NEL = 220528, negative, meaning the mesh is in the `.re2`.
- No history points.
- The ICs are "C Default"; `useric` copies the sponge arrays (§2).

### 1.2 Parameter-to-variable mapping

`uprm_read_MA` at `C/uparam.f:200-270` maps:

| Variable | Source |
|---|---|
| `UPRM_PRB` | p69 |
| `L2FREQ` | p70 |
| `FIXGEOM` | p71 |
| `NEW_DT` | p72 |
| `CHKPTSTEP` | p75 |
| `IFCHKPTRST` | p76>0 |
| `rt_kut` | INT(p78) |
| `rt_kai` | p79 |
| `rt_wght` | p80 |
| `rt_ifboyd` | (p81>0) |
| `stat_comp` | p87 |
| `stat_outp` | p88 |

- The namelist (`.upar`) readers are commented out, for example `C/rt_filter.f:36-45`.
- `uprm_read_MA` is called at every `userchk` call (`C/naca_wing.usr:357`).

### 1.3 Hard-coded physics constants

**`C/inc_src/DRL:13-18,45-47`.**
- `ctrlxs=0.25`, `ctrlxe=0.86`, `ctrlys=0.0`, `ctrlye=0.6`.
- `ypctrl=15`.
- `nuctrl=1/200000`: hard-coded Re 200k, consistent with the true Re. **FLAG:** it is inconsistent with the `.rea` p002 and the SIZE comment, which say 400k.
- `NFLDC=2`, `nfeat=5`.
- `obs_zavg = rwd_zavg = act_zavg = .true.`; all x-averaging flags are `.false.`.

**`X2Utau`** (`drl/drl_init.f:480-523`).
- A degree-9 polynomial u_tau(x). Comment: "Statistics obtained from A Tanarro … NACA0012, 2024.04.29", suction side.
- `pl(1..10)` = 0.04327887538008919, 0.5248714861235598, -4.871857463913461, 22.46151360065518, -61.58950584522501, 104.56101035004765, -108.79482929066342, 65.14084910886064, -18.9743110707296, 1.5255420287605719.
- Whether the Tanarro statistics are at Re 200k or 400k is UNKNOWN.

**Config-only values** (`HF/environment_config.yaml`):

| Key | Value |
|---|---|
| `nproc` | 4096 |
| `ndrl` | 40 |
| `znmf_avg` | 1 |
| `dt` | -1.5e-5 |
| `viscosity` | -200000 |
| `READCHKPT` | 1 |
| `CHKPFNUMBER` | 6 |
| `CHKPINTERVAL` | 5000 |
| `IOSTEP` | 5000 |
| `writeLA2` | 5000 |
| `nb_interactions` | 2500 |
| `ctrl_min/max_amp` | ±1 |
| `rescale_actions` | true |
| `u_tau` | 0.048 |
| `dUdy` | 480.54 |
| `xnel`/`Nx` | 10024 |
| `Nz` | 22 |
| `Lx` | 10 |
| `Ly` | 2 |
| `Lz` | 0.1 |

**FLAG:** `Lx: 10` is inconsistent with the mesh, which has x ∈ [-2, 4] (§6).

**Derived quantities.**
- DRL interval = ndrl·|dt| = 40 × 1.5e-5 = **6.0e-4 c/U∞**.
- Episode = 2500 × 6e-4 = **1.5 c/U∞**.
- With u_tau = 0.048 and ν = 5e-6, Δt+ per action = 6e-4 × 0.048² / 5e-6 ≈ 0.28.

---

## 2. Boundary conditions

### 2.1 Mesh BCs

From `/tmp/re2_bc.py` on `CC/naca_wing.re2`:
- Header `#v002 220528 3 220528`.
- Bounding box x ∈ [-2, 4], y ∈ [-2, 2], z ∈ [0, 0.1].
- 11308 curved edges, all of type `m` (midpoint).
- One BC section with 34876 faces:

| BC | Faces | Location / meaning |
|---|---|---|
| `P  ` | 20048 | faces 5/6: periodic in z |
| `W  ` | 5720 | 260 per z-layer, faces 1/3/4, x ≤ 0.995: wing surface |
| `v  ` | 7876 | far field |
| `o  ` | 1232 | x ≈ 3.91-3.93 element centroids: outlet plane x = 4 |

- No `v`/`o` faces lie near the wing.
- There are 22 z-layers of 10024 elements each, numbered strictly layer by layer: 0 violations of `iii=(ieg-1)/10024`. This is required by `userf`, `tripf` and `get_exyz_usr` (`drl/drl_state.f:671-697`).

### 2.2 `userbc` (`C/naca_wing.usr:170-270`)

**All user faces (`v`, `o`, and the switched wall faces).** Lines 206-213 set `pa=0`, `U0delta=1`, and `ux, uy, uz = usponge, vsponge, wsponge` at the node.

**Far field (`v  `).** Dirichlet with the **frozen field** `usponge` (§2.4).

**Outlet (`o  `), Dong-type** (lines 215-225):
- `S0 = 0.5*(1 - tanh(ux/U0delta))` and `pa = -0.5*|u|²*S0`.
- **FLAG:** `ux` here is the frozen `usponge` value, not the instantaneous velocity. With U0delta = 1 and u ≈ 1, S0 ≈ 0.119 and `pa ≈ -0.06`, a fixed nonzero, time-independent Dirichlet pressure.
- The real Dong BC uses the instantaneous u·n with a small δ (≈0.05-0.1). The NekRS port must reproduce `p = -0.5|u_frozen|²·S0(u_frozen)` to match. It must **not** use NekRS's built-in Dong outflow.

**Actuated wall faces** (lines 230-265, `#ifdef DRL`):
- Active if x ∈ [ctrlxs-0.1, ctrlxe+0.1] and y ∈ [ctrlys, ctrlye]. This is suction side only.
- Calls `ACTUATE_JET(vf, isfind, …)` (`drl/drl_action.f:167-201`). That routine sets `isfind = (msk_act(ix,iy,iz,iel) ≠ 0)` and `vf = ACTIONS(ix,iy,iz,iel)`.
- If `isfind`: `ux = vf*body_cos`, `uy = vf*body_sin`, `uz = 0` (lines 254-259). The comment says "Negative <=> Blow; Positive <=> Suction".
- Otherwise the face keeps `usponge`. At the wall that is the restart-file wall velocity, about 0.
- **How `userbc` reaches the wall faces.** `register_wall_pts` switches the CBC of the selected `W` faces to `'v  '` (`drl/drl_init.f:134`). The routines `switch_BC_2Wall` and `switch_BC_2Drichlet` (`drl/drl_reward.f:290-368`) are never called.

**UNKNOWN: shared-node conflicts.** Neighbouring agent faces share element-edge nodes and can carry different values. How Nek5000 core resolves these is UNKNOWN; `bcdirvc` is not in the allowed sources.

### 2.3 Initial conditions

`useric` (`C/naca_wing.usr:278-305`) sets `u = usponge`. In practice the solution always comes from the restart (§7).

### 2.4 Freezing of the sponge/BC field

- At `istep==0`, after `checkpoint` has loaded the restart: `opcopy(usponge, vsponge, wsponge, vx, vy, vz)` (`C/naca_wing.usr:378-382`).
- **FLAG: the frozen field is the restart snapshot of each run.** It is not a fixed reference field. Every relaunch freezes the far-field Dirichlet values and the outlet pressure from whichever restart file is loaded.
- The far-field values therefore contain turbulent and wake content wherever the restart has it. For example, the outlet at x = 4 sits inside the wake, so `pa` varies in y and z.
- With checkpoint sets 0 and 1 alternating, different restarts give different BC fields.
- NekRS port: store the istep-0 velocity in a device array and use it in `udf` BCs.

---

## 3. `userf` forcing assembly (`C/naca_wing.usr:30-147`)

**Indexing (lines 69-72).**
- `iii = (ieg-1)/nnelx1x2(1)`, the z-layer index, 0..21.
- `fzt_u = fzt1(iz + iii*lz1, 1)` and `fzt_d = fzt1(iz + iii*lz1, 2)`.

**Branch `x ≥ 6` (sponge, lines 84-101).**
- Parameters: s1start=4, s1end=10, s1rise=0.6, s1fall=0.5, s1max=1.
- `xi1 = (x-4)/0.6`, `xi2 = (x-10)/0.5 + 1`, `lmbda = s1max*(step(xi1) - step(xi2))`. `step` is defined at `C/misc.f:9-25`.
- Forcing: `ffx = lmbda(usponge-ux)`, `ffy = lmbda(vsponge-uy)`, `ffz = lmbda(0-uz)`.
- **FLAG: the sponge is dead code.** The mesh ends at x = 4, so `x ≥ 6` never holds. Even inside the domain, the rise of `lmbda` is at x ∈ [4, 4.6], so the forcing would jump from 0 to 1 at x = 6. Any HydroGym or paper statement about a sponge does not apply to this run.

**Else branch (trip, lines 116-119).**
```
ffy = exp(-((x-xup)/radiusx)**2 - ((y-yup)/radiusy)**2)*fzt_u
    + exp(-((x-xlo)/radiusx)**2 - ((y-ylo)/radiusy)**2)*fzt_d
```
- The rotated elliptic form (`alpha_elipse`) is commented out, so `alpha` from `forparam.i` is unused.
- The trip forcing is purely in the **y** direction. It is not along the wall normal.

**Final assembly (lines 135-142).**
```
if ((ffy .LE. -1.0e-10) .OR. (ffy .GE. 1.0e-10)) then
   ffy = ffy            ! trip only
else
   ffy = rtfy(ix,iy,iz,iel)
endif
ffx = rtfx(ix,iy,iz,iel)
ffz = rtfz(ix,iy,iz,iel)
```
- **FLAG: x and z forcing is the RT term only.** `ffx` and `ffz` unconditionally **overwrite** the sponge branch. So even if the sponge were reached, only its y component would survive, and only where |ffy| ≥ 1e-10.
- **FLAG: the RT term is dropped in y near the trips.** Wherever the trip Gaussian times the amplitude exceeds 1e-10 in magnitude, the y RT term is **replaced**, not added. That covers a few radii around (0.1, ±0.0468), all z, at all times except where fzt happens to cross zero.
- NekRS port: reproduce this exact either/or, not a sum.

---

## 4. RT filter (`C/rt_filter.f`) and the NekRS HPF-RT mapping

**When it is computed.**
- `make_RTF` is called from `userchk` every step (`C/naca_wing.usr:384`).
- It is built from the **end-of-step** velocity u^n and used by `userf` in step n+1. It is an explicit, one-step-lagged forcing.
- It returns zeros if `rt_wght == 0` (lines 111-116).

**Filter (`make_fil_new`, lines 375-440, `newfilter=0`).**
- diag = I, `k0 = nx - kut = 6`. For k > k0: `amp = wght·(k-k0)²/kut²` and `diag = 1 - amp`.
- For nx = 8, kut = 2, wght = 1: the modal transfer function is 1 for modes 0-5, **0.75 for mode 6 (k=7), 0 for mode 7 (k=8)**.

**Operator (`build_RT_MAT`, lines 244-301).**
- `op = V·diag·V⁻¹`, with V from `spec_coeff_init` (lines 444-535): Legendre P_k at the GLL points.
- `rt_ifboyd = .false.`, so the plain Legendre basis is used, **not** the Boyd φ_k = P_k − P_{k-2}.

**Forcing (`build_RT`, lines 305-370).**
- `rtf = (I − F⊗F⊗F) u` per element, tensor-product, unassembled (no dssum).
- `make_RTF` lines 134-136 then multiply by `rt_kai = -2000`.
- Result: **f_RT = −2000·HPF(u)**, with HPF modal weights (0, …, 0, 0.25, 1) on modes 6 and 7.

**NekRS mapping.** `regularization = hpfrt + nModes = 2 + scalingCoeff = 2000`. NekRS uses f = −χ·HPF(u).

Items to verify in the NekRS source (not in the allowed directories, so UNKNOWN here):
- **(a) Basis.** Does NekRS build the HPF Vandermonde on the Boyd-modified basis, as the Nek5000 core filter does, or on plain Legendre? If it uses Boyd, the operator differs from this case's (`ifboyd = .false.`).
- **(b) Transfer function.** Is it exactly ((k−k0)/nModes)² for the top nModes modes?
- **(c) Time integration.** NekRS evaluates the forcing on extrapolated or current u; here it is u^n, lagged explicitly. This is a minor difference.
- **(d) Trip region.** The y-component replacement in the trip region (§3) is not expressible through `hpfrt`. It must be done in `udf`, either by computing the RT term manually or by subtracting the y part.

---

## 5. Trip forcing (`C/tripf.f`, `C/forparam.i`, `C/naca_wing.wall`)

**Wall file** (`readwallfile`, `tripf.f:111-184`):
- `nwalls = 2`, `nwallpar = 5`, `npwallpar = 1`.
- `wallpar` = (tamps = 0.0, tampt = 10.0, tdt = 0.004, num_modes = 60, 0.001 "dstar", unused).
- The two walls are lines along z at (0.1, 0.0468) and (0.1, −0.0468). The per-wall `pwallpar` (0.004, 0.001, 0) is read but unused.

**Geometry** (`readtrip_par`, `tripf.f:486-525`, reads `forparam.i` at istep 0):
- xup = 0.1, yup = 0.0468, xlo = 0.1, ylo = −0.0468, radiusx = 0.004, radiusy = 0.001, alpha = 20 (unused).
- **FLAG: dead reads.** It also reads tdt = 0.004 and tampt = 10.0 into **local variables that are never used**. The live values come from `.wall`. They happen to be equal.

**z grid** (`znekgen`, `tripf.f:186-384`):
- Finds the GLL point closest to the trip line in x-y, collects the z values of elements containing it, sorts and deduplicates them.
- `kpts = 22·lz1 = 176`. `nnelx1x2 = nelgv/22 = 10024`.
- `zlength = zvec(last) − zvec(1)`.
- **UNKNOWN:** whether deduplication leaves exactly 176 z values. With 7 unique nodes per element, 22·7 + 1 = 155 would be expected. The indexing `iz + iii*lz1` assumes lz1 = 8 slots per layer, so element-interface z values must be stored twice. Not verified at runtime.

**Random signal** (`rand_func`, `tripf.f:396-440`):
- Σ_{m=1..60} sin(2π m z / zlength + 2π·ran2), **unnormalised**.
- rms ≈ √30 ≈ 5.5. Times tampt = 10, that is ≈ 55 rms and up to 600 peak.
- `ran2` (`tripf.f:443-484`) is Numerical Recipes ran2 with an LCG-type shuffle (m = 714025, ia = 1366, ic = 150889, 97-entry table). The initial seed is −32 (`naca_wing.usr:512-513`), and one `ran2` call is consumed in `usrdat3`.

**Time blending** (`tripf.f:6-79`):
1. At istep == 1: generate `fzt2 = tamps·rand = 0` and set `ntdt = −2`.
2. Loop `i = ntdt+1 .. int(time/tdt)`: `fzt3 = fzt4`, then a new `fzt4 = tampt·rand`.
3. Then `ntdt = int(time/tdt)`, `p = (time − ntdt·tdt)/tdt`, `b = p²(3−2p)`.
4. `fzt1 = fzt2 + (1−b)fzt3 + b·fzt4`.

**FLAG: the trip realization depends on restart history.**
- `tripf` is also called at istep 0 (`naca_wing.usr:431`). There `ntdt` is the common-block initial value 0, so the loop already runs i = 1..int(t0/tdt), which is ≈ 7050 iterations × 2 walls × 60 `ran2` calls for t0 = 28.2.
- At istep 1 it resets `ntdt = −2` and regenerates from i = −1, continuing the `ran2` stream.
- The trip forcing is therefore a deterministic function of (t0, t), not of t alone. Different restart times give different trip histories.
- Bit-exact reproduction in NekRS requires replicating this call sequence and `ran2` exactly. A statistically equivalent trip needs only: 60 random-phase spanwise modes, amplitude 10, a new realization every 0.004 c/U∞ with smoothstep blending, and Gaussian radii 0.004 × 0.001.

`ARnumnodes` is never called.

---

## 6. Geometry fixes

**`usrdat2`** (`C/naca_wing.usr:465-492`):
- `FIXGEOM = INT(PARAM(71)) = 0`, so `fix_mygll` (`C/misc.f:28-115`) is **inactive**.
- `call fix_geom` (line 489) runs **always**. This is the core Nek5000 `fix_geom` (smoothing of GLL points / dssum averaging).
- NekRS has no direct equivalent. **UNKNOWN** whether it changes anything for this mesh; it is likely a no-op for a conforming curved mesh.

**FLAG: `fix_mygll` uses the wrong airfoil file.**
- It reads `naca4412.dat` (1000001 points) for a NACA0012 case.
- It snaps `W` GLL points with x < 0.99 to the nearest point, then does Laplacian mesh blending (`my_poss_soln`: h1 = 1 + 9exp(−d/0.25), tol 1e-12) and outposts `gri`.
- It is inactive, but would be wrong if enabled. The `naca4412.dat` file is not in the case directories.

**Surface definition.**
- Curvature comes only from `m` (midpoint) curved edges in the `.re2`.
- NekRS needs the `.re2` converted with curvature honoured: the high-order GLL coordinates. The safest route is to use the coordinates stored in the restart/`.f` file, if any (the `XUP` restart has no X; §7), or `nekrs --build-only` with the `.re2`.
- UNKNOWN whether NekRS's midpoint-curve handling reproduces Nek5000's `fix_geom`-smoothed geometry to round-off.

**Domain.** x ∈ [−2, 4], y ∈ [−2, 2], z ∈ [0, 0.1] (span 0.1c, periodic). That is only 2c upstream and 3c downstream: a small domain with Dirichlet frozen-field far-field BCs (§2).

---

## 7. Restart / checkpoint and file format

**Scheme** (`C/chkpoint.f`, Peplinski full restart, `CHKPTNRSF = 4`):

*Set selection.*
- `checkpoint_init` reads the set number from `naca_wing.restart` (content `1` in both `C/` and `HF/`).
- Only 0 or 1 is accepted (`chkpoint.f:213`).

*File names.*
- `rs8naca_wing0.f000(4·set + i)` for i = 1..4, built by `IO_mfo_fname` in `IO_tools.f` with prefix `rs8`, session name, and `0.f` (MPIIO, so ndigit = 1).
- Set 1 maps to `f00005`-`f00008`; set 0 maps to `f00001`-`f00004`.

*Reading.*
- `checkpoint_IO`: for ISTEP < 4 it calls `checkpoint_pert`, which for DNS calls core `restart(1)` with `initc = file(istep+1)` (`chkpoint.f:565-588`).
- So files 1-4 are loaded at istep 0, 1, 2, 3, each **overwriting** the solution. This yields an exact BDF3/EXT3 history after 3 steps.
- **The files carry no lag fields.** The history is rebuilt by re-reading.

*Writing.*
- `restart_save_pert` writes `XUP` via `outpost2` (`chkpoint.f:525`) with `wdsizo = 8`.
- It writes when `istep > iosav/2 and mod(istep + iosav, iosav) < 4`: four consecutive steps around each multiple of 5000.
- `naca_wing.restart` toggles 0/1 after each save.

*UNKNOWN items.*
- The config has `READCHKPT 1` and `CHKPFNUMBER 6`. How HydroGym maps `CHKPFNUMBER 6` to `naca_wing.restart` or to file names is UNKNOWN.
- Under the Fortran logic, 6 is neither a valid set (0/1) nor a file count (4).

**File format.** Probe of `HF/restart_files/init_1/rs8naca_wing0.f00001`, 21353285320 bytes:

*Header and fields.*
- Header `#std 8 8 8 8 220528 220528 0.2819999999926E+02 170000 0 1 XUP`.
- 8-byte words; N = 7 (8³ points per element); 220528 elements; t = 28.2; step 170000; `XUP` = coordinates, velocity, pressure.
- Then float64 fields, then float32 per-element min/max metadata (56 bytes per element).

**FLAG: stale tail.** The rest of the file is **stale bytes of an earlier N = 11 (lx = 12) `XUP` float64 file** with the same nel. It was overwritten without truncation, since MPI-IO does not truncate.
- Evidence: file size = 136 + 4·nel + 7·nel·1728·8 + 7·8·nel exactly.
- The character of the data changes at the old field boundaries (≈3.11F for U, 13.24F for P).
- NekRS (or a converter) must read only the N = 7 part and ignore the tail. A reader that infers size from the file length will fail.

**Flow state** (README / analysis log):
- The restart is a Re = 200k flow with U∞ ≈ 0.997.
- Cf(0.65) ≈ 5.7e-3.
- The boundary layer is transitional in the control window: minimum dU/dn ≈ 330 at x/c ≈ 0.3.

**Mask file.** `mask_naca_wing0.f00002` is required by `calculate_norm_angle` (§8.2). It is an HF file of 3.6 GB that has not been downloaded.

**NekRS port.**
- Read 4 consecutive N = 7 fp64 restart files into the BDF history. Alternatively, accept a first-order startup and document the deviation.
- NekRS reads only one file natively. Reproducing the exact BDF3 history needs custom `udf` code that injects the lag fields, or 3 extra steps with re-reads.
- Write outputs as fp64.

---

## 8. DRL coupling (compared with `channel_case/DRL_SPEC.md`)

### 8.1 Control loop (`drl/drl_main.f`) and MPI

**Request structure.** Identical to the channel:
- ISTEP 0: `MPI_INTERCOMM_CREATE` (tag 99), then `drl_init`.
- Afterwards rank 0 receives a 5-character request (tag 22) and broadcasts it.
- Requests: `STATE`, `CNTRL`, `TERMN` (`exitt0`; prints "CKPT SAVED" but writes none), `RSETS` (only sets a flag that is never read), `EVOLV`, `INTAL`.
- `drl_chkpt_reset` is never called.

**Hold length.** `drl_step = int(PARAM(89))` (line 37), so the hold length is p089.
- **FLAG: p089 = 0 in the shipped `.rea`.**
  - The test `i_evolv.ne.drl_step` never becomes false, since `i_evolv` starts at 1 and increments.
  - Nek would then evolve forever after the first `EVOLV` and never send a reward (`drl/drl_IO.f:50`). Python would deadlock.
  - So HydroGym must write p089 = ndrl = 40 for the case to work at all. That it does is UNKNOWN from these sources (inferred only).
  - The channel takes ndrl from `uparam(1)` in `.par`; the wing uses the `.rea` p089.
- **Per cycle:** 40 steps under constant actions. Policy steps are at ISTEP = 1, 41, 81, ….
- **No smoothing:** `moving_smooth_action` is commented out (`drl_main.f:149`). `exp_smooth_action` is never called.

**Messages** (`drl/drl_IO.f`). Same tags as the channel:
- **INTAL:** `node_list(LP)` (tag 1996); per rank, NUMCTRL plus `info_agt(1..5)` (tags NID + 10000·k) and `pos_agt` x, y, z (tags NID + 100000·k).
- **STATE:** TIME (1998), |COURNO| (1999), then 2 fields (tag `NID*il + 70000`).
- **CNTRL:** TOTCTRL = 5000 doubles per rank (tag NID + 90000).
- **Reward:** CFL (1999) every evolve step, and `rwd_agt` (NID + 80000) at `i_evolv == drl_step`.

### 8.2 Agents (`drl/drl_init.f`)

**Registration** (`register_wall_pts`, lines 48-169):
- Selects `W` faces whose GLL face midpoint satisfies x ∈ (0.25, 0.86) and y ∈ (0, 0.6). This is **suction side only**.
- Registers all 64 face GLL nodes via `facind`, duplicates included (the same as channel `gll_unique = 0`).
- Stores `info_agt = (lglel, iface, ix, iy, iz)`, sets CBC to `'v  '` (line 134), and exits if NUMCTRL > TOTCTRL.
- `sort_wall_pts` bubble-sorts by x within each rank.

**Counts** (from the `.re2`, `/tmp/wall_agents.py`, vertex-midpoint test):
- **1408 faces** (64 per z-layer × 22), giving **90112 registered node copies**.
- x range of face midpoints: 0.254-0.852. No faces lie within 2e-3 of the x limits.
- Unique nodes ≈ (64·7 + 1) × (22·7) = 449 × 154.
- After z-averaging (all of obs, act and rwd are z-averaged) there are at most 449 distinct x-stations (UNKNOWN how Python groups them into agents).

**Normals** (`calculate_norm_angle`, lines 282-371):
- Loads `mask_naca_wing0.f00002` via `load_fld` and uses its vy, vz fields as a reference point (xnn, ynn) per node. The meaning of this reference point is UNKNOWN; the file is not available.
- `bdx = |x − xnn|`, `bdy = |y − ynn|`, `r = √(bdx² + bdy²)`, `body_sin = bdy/r`, `body_cos = bdx/r`.
- `body_cos` is negated only in case B (x ≤ xnn and y > 0).
- It runs in `usrdat3`, before the restart read. The vx/vy/vz it overwrites are then replaced by the restart.
- **The NekRS port needs this file or an equivalent analytic normal.** That gives (cos, sin) = n̂ up to sign; whether n̂ points into the fluid is UNKNOWN.

### 8.3 Actions (`drl/drl_action.f`)

**Receiving.** `recv_Actions` (lines 51-163) receives TOTCTRL doubles and scatters `ACTIONS(ix,iy,iz,iel)`. It sets `msk_act = 1` only if ZNMF is off (p090 ≤ 0).

**ZNMF** (`znmf_avg`, lines 207-333), active because p090 = 10 > 0:
1. **u_tau rescaling** (`#ifdef UTAU`, lines 243-258): `a ← a · X2Utau(x_agent)`. The Fortran **does** scale here, unlike the channel.
2. **Spanwise mean removal:** z-average per (ix, iy, 2D element) using `wzm1` weights and global `gop`, via `z_averaging`/`z_avg_reshape` (`drl/drl_state.f:431-549`). The final `ACTIONS = a·u_tau − ⟨a·u_tau⟩_z`.
   - **FLAG:** the switch used is `rwd_zavg`, not `act_zavg` (line 268). Both are `.true.`, so this does not change behaviour.
   - **FLAG:** the `rwd_xavg` branch calls `z_averaging` with `avgVX` (lines 277-279). This is a bug, but inactive.
   - No dssum is applied, unlike the channel. Each element-local copy is averaged with the GLL weights `wzm1(k)`, which is exact quadrature along z.
   - So the zero-net-mass-flux constraint is **per spanwise line at each x-station**, not over the whole wall.
3. `msk_act = 1` is set on every node of every agent face (`impose_ivalue`, lines 290-291, 504-526).

**Imposed BC.** `u_wall = (a·u_tau(x) − ⟨·⟩_z)·(body_cos, body_sin)` along the body normal.
- The magnitude per unit action is ≈ u_tau(x) ≈ 0.03-0.05.
- Python `rescale_actions` and `ctrl_*_amp ±1` act before this (UNKNOWN exactly how).

**Comparison with the channel.**
- The channel imposes `uy = dssum(a) − ⟨⟩` with no u_tau factor and with double counting.
- The wing has no dssum and therefore no double counting, but does apply u_tau(x) scaling.

### 8.4 State (`drl/drl_state.f`)

**Sensing points** (`interp_sensing_plane`, lines 376-473):
- Normal offset `ynorm = 15·nuctrl/u_tau(x)` from each wall node.
- Point = wall ± ynorm·(cos, sin), with + if y_w > 0 and − otherwise.
- Located with `findpts` (`register_ctrl_pts`).

**Fields** (`vel_fluct_compute`): z-mean fluctuations of u and v (obs_zavg) using the same `z_averaging`.

**Projection** (`normal_V_project`, lines 116-174):
- `Ut = vx·body_sin + vy·body_cos` and `Vn = vx·body_cos + vy·body_sin`.
- **FLAG: this is not a rotation.**
  - With n̂ = (cos, sin), the tangent should be ±(−sin, cos), i.e. ±(−vx·sin + vy·cos).
  - Both components use + signs, so Ut·Vn are not orthogonal projections. For example, at θ = 45° both equal (vx + vy)/√2.
  - It also outposts `rot` at ISTEP == 1.

**Scaling** (`vel_pts_find`):
- Under UTAU, each field is divided by `X2Utau(x) + 1e-15`.
- **FLAG:** the scaling loop runs over TOTCTRL instead of NUMCTRL. This is harmless: garbage entries are scaled but unused.
- Unlike the channel, the state is normalised by u_tau(x) in Fortran.

### 8.5 Reward (`drl/drl_reward.f:47-208`)

**Computation.**
- `gradm1(vx)` gives dU/dx and dU/dy, which are then z-averaged (rwd_zavg).
- The rotation at lines 155-158:
  ```
  dUdy = -body_cos*body_sin*dUdx - body_cos*body_cos*dUdx
         + body_sin*body_sin*dUdy + body_cos*body_sin*dUdy
  ```
- The comment says "remember body_sin == cosTheta".
- **FLAG: not a tensor rotation.**
  - Only ∂u/∂x_j is used; ∂v/∂x_j is not, so this is not ∂u_t/∂n.
  - The coefficients do not form a rotation; for example, at θ = 0 the result is −dUdx·1 + 0.
  - The quantity is some ad-hoc mix whose relation to wall shear (Cf) is unclear except where the wall is nearly horizontal and body_cos ≈ 0.

**Per-agent value.** `rwd_i = |dUdy_rot|` at the agent wall node. It is local per node, whereas the channel uses the global wall mean.

**Running mean.**
- `rwd = x1` at i = 1, then `(rwd·i + x_i)/(i+1)`.
- **FLAG: the same mis-weighting as the channel.** Sample 1 is double-weighted, giving (2x₁ + x₂ + … + x₄₀)/41.
- The sample window is offset by one step from the action window.

**Python side.** Normalisation by `dUdy = 480.54` and the reward formula (UNKNOWN beyond the config).

**FLAG: header comment.** The file header says "TODO: Need to be eventually adopted to the wing" (line 3).

### 8.6 Opposition control

`oppo_ctrl/*.f` and `inc_src/OPPO_CTL` are not compiled (§0). Note that `oppo_ctrl.f:815-959` also defines `z_averaging` and `get_exyz_usr`, which would clash with `drl_state.f` if it were linked.

---

## 9. Statistics and outputs; Cf(x), Cd,f, Cd

**2D statistics** (KTH, `stats/statistics_2D.f:75-112`, `stat_avg_all`):
- Homogeneous direction z (`STAT_IDIR = 3`, `statistics_2D_usr.f:47`).
- Initialised at istep 0. Sampled every `stat_comp = 10` steps.
- Collated and written to `sts*` files (prefix `sts`, `statistics_2DIO.f:97`) every `stat_outp = 5000` steps.
- **The average resets after each write** (`STAT_ATIME = 0`, lines 104-105).
- Variables:
  - the standard set of up to 60 quantities: U, V, W, P, second- to fourth-order moments, pressure-strain, dissipation tensors, etc. (`statistics_2D.f` from about l.1265);
  - plus 6 user variables: ⟨rtfx⟩, ⟨rtfy⟩, ⟨rtfz⟩, ⟨rtfx·u⟩, ⟨rtfy·v⟩, ⟨rtfz·w⟩ (`statistics_2D_usr.f:260-320`, npos 61-66).
- `STAT_LVAR = 72` (`inc_src/STATS`).

**Point time series** (`stats/time_series.f`):
- Initialised from `stat_pts.in` in `stat_init` (`statistics_2D.f:215-218`) and sampled every `5·stat_comp = 50` steps (`statistics_2D.f:1289-1292`).
- Buffered `istcount = PARAM(69) = 10` snapshots.
- `CC/stat_pts.in`: header `2123 400 3`, 2125 lines. `LHIS = 220` points per rank.
- **FLAG:** `stat_pts.in` is present in `CC/` but **absent from `C/` and `HF/`**. Whether the HydroGym run has it, and what `stat_pts_read` does when it is missing, is UNKNOWN. Line 103 opens it with `err=100`, which jumps to an error exit.

**Other outputs.**
- `la2` (velocity plus lambda2 in T) every `L2FREQ` steps (`naca_wing.usr:389-404`).
- `rs8` checkpoints (§7).
- `rot` at ISTEP 1 (§8.4).

**Cf(x), Cd,f, Cd: there is no Fortran computation.**
- `grep` for `torq`, `drag`, `lift`, `tau_w` and skin friction in `C/*.f`, `C/naca_wing.usr` and `C/drl/*.f` finds nothing relevant. There is no `torque_calc` or `drag_calc` call.
- Cf(x) and the drag coefficients reported in the paper must have been computed **offline**, from `sts` files or snapshots, or in HydroGym Python. The script and the averaging window are **UNKNOWN**.
- Because the `sts` averages reset every 5000 steps (0.075 c/U∞), any longer average must be assembled in post-processing.
- The only on-line wall quantity is the per-agent reward of §8.5. That is not Cf: it is an ad-hoc gradient combination, z-averaged, over the suction-side window only.

---

## 10. NekRS port checklist and open questions

### Checklist
1. **Mesh.**
   - Convert `CC/naca_wing.re2` (220528 hex elements, N = 7, i.e. `polynomialOrder = 7`). Curvature is midpoint `m`.
   - Boundary IDs: wall `W`, far-field `v`, outlet `o`, z-periodic `P`.
   - Keep the layer-by-layer element numbering, or replace every `(ieg-1)/10024` index with a coordinate-based z-layer and x-station lookup.
2. **Physics.** ρ = 1, ν = 1/200000 (not 400k), fixed dt = 1.5e-5, BDF3/EXT3, PN-PN (NekRS) instead of PN-PN-2 (a discretisation difference to document), dealiasing on.
3. **Tolerances.** Velocity 1e-8, pressure 5e-6 in the `.rea` vs 1e-6 in the config (UNKNOWN which applied).
4. **Stabilisation.** `hpfrt`, nModes = 2, scalingCoeff = 2000. Verify that the basis is not Boyd (§4). Handle the y-replacement in the trip region in `udf` (§3).
5. **Trip.** In `udf` source: y-direction Gaussians at (0.1, ±0.0468) with radii 0.004 × 0.001, 60 random-phase z modes, amplitude 10, Δt_trip = 0.004, smoothstep blending. For exactness, port `ran2` and the istep-0/istep-1 call sequence (§5).
6. **Sponge.** None in effect. Do not add one, or document it if you do (§3).
7. **BCs.**
   - Freeze u at step 0 from the restart into a device array.
   - Far field: Dirichlet u_frozen.
   - Outlet: Dirichlet p = −0.5|u_frozen|²·0.5(1 − tanh(u_frozen,x)), with zero-gradient velocity (§2.2).
   - Wall: no-slip, except the actuated suction-side faces x ∈ (0.25, 0.86), which get u = a_ZNMF·n̂ (§8.3).
8. **Restart.**
   - Read only the N = 7 fp64 part of `rs8naca_wing0.f0000{5..8}` (set 1) or `{1..4}`, ignoring the stale tail.
   - Reproduce the 4-file BDF history or document a cold start.
   - Initial time 28.2, step 170000.
9. **DRL bridge.**
   - Reproduce the MPI protocol and tags, ndrl = 40, 90112 registered node copies (with duplicates) and the x-sorted order per rank. Agent ordering depends on the rank decomposition, which differs in NekRS, so check what Python assumes.
   - State: (Ut, Vn) as defined (with the suspect projection), z-fluctuations at the y+ = 15 offset point divided by u_tau(x).
   - Action: u_tau scaling plus z-mean removal.
   - Reward: |ad-hoc rotated dU/dy|, z-averaged, with the mis-weighted running mean.
10. **Normals.** Obtain `mask_naca_wing0.f00002` or derive equivalent (cos, sin) values analytically. Preserve the sign conventions, including the case-B negation.
11. **Statistics.** z-averaged 2D statistics every 10 steps, output and reset every 5000 steps. Provide Cf(x) and Cd offline with a documented script and window.

### Open questions (UNKNOWN)
- Q1. Does HydroGym write p089 (it must for the run not to deadlock), p090, p011, p015, p021 and p070, and with what values?
- Q2. How does `CHKPFNUMBER: 6` map to `naca_wing.restart` (0/1) or `rs8` file numbers? Which restart set was used for the paper's wing evaluation?
- Q3. What do the vy/vz fields of `mask_naca_wing0.f00002` contain? Does n̂ point into the fluid, and what is the actual sign of "suction"?
- Q4. Which offline script and averaging window produced the paper's Cf(x), Cd,f and Cd? Is the control-window Cd,f reduction computed from the `sts` mean or from snapshots?
- Q5. How are the 90112 node copies grouped into agents on the Python side, and how is the zero-shot channel policy's input normalised for the wing (`u_tau 0.048`, `dUdy 480.54`) on top of the Fortran u_tau(x) scaling?
- Q6. NekRS `hpfrt` basis (Boyd vs Legendre) and transfer function (§4).
- Q7. How does Nek5000 core resolve Dirichlet values that differ between neighbouring agent faces on shared nodes?
- Q8. Was `stat_pts.in` present in the HydroGym run?
- Q9. Are the Tanarro u_tau statistics behind `X2Utau` at Re 200k or 400k?
- Q10. Is the deduplicated trip z-grid count 176 or 155 (§5)?
