---
sidebar_label: wall_jets
title: hydrogym.maia.wall_jets
---

Actuation power of wall jets of m-AIA's boundary condition 2007 on general walls (flat faces, airfoils, spheres).

`hydrogym.maia.power.SlotJetPower` covers slot jets on circular cylinders. This module covers the other jet
geometries of BC 2007 (`bc20007` in `src/LB/lbbndcnddxqy.cpp`): flat plate (`lbObjectClass` 1, e.g. the open
cavity and the NACA 0012 airfoil), cube (2), sphere (3) and the circular-body rule (0) on flat faces (square
cylinder, cavity cases with `lbJetRadius` 0).

**Jet geometry (as the solver does it).** `Bc2007Jets` reproduces the tagging of BC 2007: a boundary cell
belongs to jet `j` if its centre lies in the jet's bounding box (`lbJetCenter` +- `lbJetRanges` / 2, cut by
`lbJetLimits`) and its distance from the jet axis is within the half range (rule per object class: distance from
the jet centre (class 1), across the slot axis (class 2), as arc length along the slot axis plus an azimuthal range
(class 3), across the jet direction (class 0)); its wall velocity is `command_j * factor` with the parabolic factor
`1 - d^2 / h^2` times the jet direction (`lbJetAngles`, `lbJetNormals` or the sphere radial). The 2D
last-box-wins tagging of the solver is kept. The factor is evaluated at the centre of the fluid cell next to the
wall (the cell whose links carry the wall velocity), not at the wall itself:

* flat walls on cell faces (cavity, square cylinder, cube): the jet consists exactly of the fluid cells adjacent to
the wall, one wall element per cell (area `dX^(nDim-1)`), factor at the cell centre (`flat_wall_jets`);
* curved walls (airfoil from the case's geometry file, sphere) and flat walls off the cell faces: the moving-wall
bounce-back is modelled link by link (`link_jets`): which boundary cell writes each cut link (the fluid cell
for `q <= 1/2`, the solid cell across it otherwise, as the solver's redundant-cut correction does), its moving-
wall source, and the flow-dependent part of the interpolated bounce-back, which `WallJetPower` evaluates
from the probed cell values. On the NACA 0012 (`c/128`, jet centres inside the airfoil) these terms are 12-36 %
of the jet mass flux; with them the model reproduces the solver's mass flux (control volume) within 4 % (within
1.2 % on the cavity walls off the cell faces, where they are up to 29 %).
* `sphere_jets` and `curve_jets` are continuous quadratures of the jet profile on the wall (cross-checks).

**Power.** `P = int v_n (p - p_ref) dA + int v_n rho |u_w|^2 / 2 dA` point-wise per wall element (Bewley, Moin &
Temam 2001; `hydrogym.power.transpiration_forms`) with `v_n = u_w . n` (`n` into the fluid, `v_n > 0`
blowing; with the interpolation terms of `link_jets` the actual mass flux per element divided by `rho dA`).
The wall pressure and density are extrapolated to the wall from the cell-centre values at `wall_distances`
(default 1.5-4.5 cells) along the wall normal (quadratic least squares; m-AIA returns probe values by nearest
neighbour, so the model probes the cell centres of the finest level itself and interpolates multilinearly); the
relative difference between the quadratic and the linear extrapolation is kept as `last_extrapolation_spread`.
`p_ref` is the area-mean wall pressure over all jets of the environment (one actuator
group). Where the jets are not zero-net-mass-flux (independent actions), the signed power then does not depend on
the pressure gauge but leaves out the work `p_ref Q` of supplying the net flow `Q` (as in the Firedrake cavity
and cylinder environments).

Normalisation: `C_P = 2 P / (U^3 L_proj)` with `U = Ma / sqrt(3)` and `L_proj` the projection length (2D, cells)
or area (3D, cells^2) of the environment's drag coefficient.

## Bc2007Jets Objects

```python
@dataclass
class Bc2007Jets()
```

Jet definition of m-AIA's BC 2007, from the solver properties (geometry units).

`centers` are the jet centres after m-AIA's shift (object class 0: `lbJetCenter + lbJetRadius (cos, sin)`),
`limits` the box offsets per jet `[-x, -y(, -z), +x, +y(, +z)]` with non-positive entries = no limit.

#### centers

(nj, nDim)

#### angles

(nj,)

#### ranges

(nj,)

#### limits

(nj, 2 nDim), +inf where unlimited

#### normals

(nj, nDim), class 2

#### slot\_axes

(nj,), classes 2 and 3

#### sphere\_center

(nDim,), class 3

#### sphere\_radius

class 3: lbJetRadius[0]

#### from\_properties

```python
@classmethod
def from_properties(cls, props: dict) -> "Bc2007Jets"
```

From the property dict of the case (`properties_run.toml`).

#### boxes

```python
def boxes()
```

Bounding boxes (lo, hi), each (nj, nDim), as in the solver (open intervals).

#### tag

```python
def tag(x: np.ndarray)
```

Jet index (-1: none) and multiplication factor vector (wall velocity per unit command) of cells at `x`.

`x`: cell centres (n, nDim), geometry units. Reproduces the kernel of `bc20007` (first call).

## WallElements Objects

```python
@dataclass
class WallElements()
```

Discretised jet walls: one entry per wall element.

points: wall points (n, nDim), geometry units; normals: unit direction of the element's flux vector (into the
fluid); weights: element area in lattice units (cells^(nDim-1)); factors: wall velocity per unit command (n,
nDim); jet: jet index (n,); inside: callable, True for points inside the solid (checks the pressure stencils);
sample_normals: wall normals along which the pressure is sampled (default: normals); links: the cut lattice links
of the elements (`link_jets`), whose interpolation terms make the mass flux flow-dependent.

#### subset

```python
def subset(mask) -> "WallElements"
```

The elements selected by `mask` (with their links).

#### flat\_wall\_jets

```python
def flat_wall_jets(jets: Bc2007Jets,
                   dX: float,
                   lattice_origin: Sequence[float],
                   span: Optional[Sequence[float]] = None,
                   max_cells_per_axis: int = 64,
                   inside: Optional[Callable] = None) -> WallElements
```

Jets on flat walls aligned with the lattice axes, exactly as the solver discretises them.

The wall of jet `j` is the plane through its centre normal to its blowing direction (object classes 0 and 1:
`lbJetAngles`; class 2: `lbJetNormals`), which must be a lattice axis. If the plane lies on cell faces, its
jet cells are the fluid cells adjacent to it whose centres pass the BC 2007 test (`Bc2007Jets.tag`); each
is one wall element of area `dX^(nDim-1)` at the cell's wall foot point, with the factor of the cell centre.
Otherwise (2D) the jet is modelled link by link (`link_jets` with the half plane as the solid).

**Arguments**:

- `span` - (lo, hi) of the domain along z (3D), used where the jet box is not limited in z.
- `max_cells_per_axis` - in 3D, along a wall direction in which the jet profile is uniform (no box limit inside the
  domain other than the extent), every k-th cell is used (with k-fold weight) so that at most this many
  remain; the profile direction is always resolved cell by cell.

#### concat\_elements

```python
def concat_elements(parts: Sequence[WallElements],
                    inside=None) -> WallElements
```

Concatenation of element sets.

#### read\_polyline\_stl

```python
def read_polyline_stl(path: str, n_dim: int = 2) -> np.ndarray
```

Wall segments (m, 2, 2) of a 2D m-AIA geometry file (first line: number of points, then x y per point).

#### read\_stl\_section

```python
def read_stl_section(path: str, z: float) -> np.ndarray
```

Section of a 3D STL surface (ASCII or binary) with the plane `z`: segments (m, 2, 2) in the x-y plane.

#### segments\_inside

```python
def segments_inside(segs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]
```

Point-in-solid test for the closed curves made of `segs` (ray casting; unordered segments are fine).

#### curve\_jets

```python
def curve_jets(jets: Bc2007Jets,
               segs: np.ndarray,
               dX: float,
               per_cell: int = 8,
               cell_offset: float = 0.5,
               z_planes: Optional[np.ndarray] = None,
               span_weight: float = 1.0) -> WallElements
```

Jets on a curved 2D wall (polyline), optionally extruded along z (3D, spanwise-uniform geometry).

The wall segments near the jets are subdivided (`per_cell` points per cell length, midpoint rule); the
normal points into the fluid (outside the closed curve); the factor is that of the cell centre `cell_offset`
cells out along the normal. In 3D the elements are repeated on `z_planes` (cell-centre planes) with weight
`span_weight` (cells of span per plane).

#### sphere\_jets

```python
def sphere_jets(jets: Bc2007Jets,
                dX: float,
                n_theta: int = 8,
                per_cell: float = 1.0,
                cell_offset: float = 0.5,
                max_phi: int = 128,
                keep_all: bool = False) -> WallElements
```

Jets of object class 3 (sphere): quadrature over the band of each jet on the sphere surface.

Across the band (polar angle about the slot axis): Gauss-Legendre over the exact support of the parabolic profile
of the cells `cell_offset` cells out along the radius; along it (azimuth): midpoint rule over the jet's angular
range (about `per_cell` points per cell, at most `max_phi`).

#### lattice\_links

```python
def lattice_links(n_dist: int)
```

Non-rest lattice velocities (q-1, nDim) and weights of D2Q9, D3Q19 or D3Q27.

#### link\_jets

```python
def link_jets(jets: Bc2007Jets, dX: float, cells: np.ndarray, inside: Callable,
              cut: Callable, foot: Callable, n_dist: int) -> WallElements
```

Jets on a curved wall, at the level of the lattice links (the moving-wall bounce-back of BC 2007).

BC 2007 imposes the wall velocity `u_W` of a boundary cell (factor of its centre, `Bc2007Jets.tag`) through
the moving-wall source `S = 2 w_k rho c_k . u_W / c_s^2` of the population it reflects into the fluid along each
cut link (`interpolatedBounceBackSingleSpecies`, Bouzidi et al. 2001). Which cell writes a link follows the
solver (`calculateWallDistancesSTD_correctRedundantCuts`): the fluid boundary cell where the link is cut at
`q <= 1/2` (source `S` with its own `u_W`), the solid boundary cell across the link where `q > 1/2`
(source `S / (2 q)` with the solid cell's `u_W`). The fluid boundary cells form one wall element each (flux
vector `F = 6 sum_k w_k c_k` of their reflected directions `c_k`), every solid-cell link one more.

The interpolation of the bounce-back adds a flow-dependent part to the mass that crosses each cut link per step:
`(1 - 2q) [f_k(x + c_k) - f_k(x)]` (fluid cell writes; populations moving into the wall, from the next cell
out) and `(2q - 1) / (2q) [f_k'(x) - f_k(x)]` (solid cell writes; `k'` the reflected direction). It vanishes for
walls on cell faces (`q = 1/2`). The links are kept in `links` so that `WallJetPower` can add this part
from the probed cell values (equilibrium populations).

**Arguments**:

- `cells` - candidate fluid-side cell centres (n, nDim) near the jets (every fluid boundary cell of the jets must
  be among them; others are ignored).
- `inside` - point-in-solid test; `cut(x0, x1)`: fraction in (0, 1] of the first wall crossing on the segment,
  `nan` if none; `foot(x)`: nearest wall point and the wall normal into the fluid.
- `n_dist` - lattice (9, 19, 27).

#### polyline\_wall

```python
def polyline_wall(segs: np.ndarray)
```

`inside`, `cut` and `foot` of a closed 2D wall made of segments (m, 2, 2).

#### lattice\_cells\_in\_boxes

```python
def lattice_cells_in_boxes(jets: Bc2007Jets,
                           dX: float,
                           lattice_origin,
                           margin: float = 2.0,
                           span=None)
```

Finest-level cell centres in the union of the jets' bounding boxes, widened by `margin` cells.

#### curve\_link\_jets

```python
def curve_link_jets(jets: Bc2007Jets,
                    segs: np.ndarray,
                    dX: float,
                    lattice_origin,
                    n_dist: int = 9,
                    z_planes: Optional[np.ndarray] = None,
                    span_weight: float = 1.0) -> WallElements
```

`link_jets` on a 2D wall made of segments; in 3D extruded along z (spanwise-uniform geometry).

In 3D (D3Q27 = D2Q9 x D1Q3) the in-plane sources of the extruded geometry equal those of D2Q9, so the section is
treated in 2D and repeated on `z_planes` (cell-centre planes) with weight `span_weight` cells each.

#### sphere\_link\_jets

```python
def sphere_link_jets(jets: Bc2007Jets,
                     dX: float,
                     lattice_origin,
                     n_dist: int = 27) -> WallElements
```

`link_jets` for the jets of object class 3 on the sphere (`lbJetCenterSph`, `lbJetRadius[0]`).

## WallJetPower Objects

```python
class WallJetPower()
```

Actuation power of BC-2007 jets given as wall elements (see the module docstring).

**Arguments**:

- `elements` - `WallElements` of all jets (from `flat_wall_jets`, `curve_jets` or
  `sphere_jets`).
- `n_jets` - number of jets (actions).
- `lattice_origin` - origin of the finest-level cell-centre lattice (cell centres at `origin + (i + 1/2) dX`).
  dX, Ma: finest cell size and Mach number (`U = Ma / sqrt(3)`).
- `l_proj` - projection length (2D, cells) or area (3D, cells^2) of the environment's drag coefficient.
- `wall_distances` - distances of the pressure samples from the wall along its normal (cells).

#### coefficient

```python
def coefficient(p_lattice)
```

Lattice power -> C_P.

#### wall\_state

```python
def wall_state(probe_data: np.ndarray, degree: int = 2)
```

Wall pressure and density per element, by least-squares extrapolation along the normal.

#### interpolation\_flux

```python
def interpolation_flux(probe_data) -> np.ndarray
```

Flow-dependent mass per step through the cut links of each element (bounce-back interpolation, see
`link_jets`), from equilibrium populations of the probed cells; zeros without links.

#### point\_terms

```python
def point_terms(probe_data, commands)
```

Pressure and kinetic power density per element and the element weights (lattice units).

#### mass\_flux

```python
def mass_flux(commands, rho: float = 1.0) -> np.ndarray
```

Nominal mass flux of each jet into the fluid (moving-wall sources only; lattice units, per unit span in
2D) for density `rho`; `last_mass_flux` adds the interpolation terms of the last evaluation.

#### evaluate

```python
def evaluate(probe_data, commands) -> dict
```

Per-jet signed C_P and the cost forms (point-wise, pressure and kinetic terms separate).

#### geometry\_files

```python
def geometry_files(geometry_toml: str, bc: int = 2007) -> List[str]
```

Geometry files of the segments with boundary condition `bc` in an m-AIA `geometry.toml`.

