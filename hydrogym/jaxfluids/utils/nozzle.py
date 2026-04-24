import os
import platform
from dataclasses import dataclass
from pathlib import Path

if platform.machine().lower() in {"aarch64", "arm64"}:
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
import json
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jax import Array
from numpy import ndarray

pv.global_theme.allow_empty_mesh = True


from jaxfluids import SimulationManager
from jaxfluids_thirdparty.gas_dynamics.core import (
    density_from_pressure_temperature,
    speed_of_sound,
    total_energy,
)
from jaxfluids_thirdparty.gas_dynamics.isentropic import (
    density_ratio_isentropic,
    mach_number_from_pressure_ratio_isentropic,
    pressure_ratio_isentropic,
)

TargetThrustAngle = Array | float
TargetThrustAngleFn = Callable[[Array | float], TargetThrustAngle]


@dataclass(frozen=True, slots=True)
class NozzleGeometry:
    """Fixed nozzle geometry based on the publication
    Das et al. 2025 AIAA
    """
    A: tuple[float, float] = (0.0,-0.01559)
    B: tuple[float, float] = (0.0, 0.0352)
    C: tuple[float, float] = (0.02329, 0.02954)
    D: tuple[float, float] = (0.05049, 0.01552)
    E: tuple[float, float] = (0.0608076, 0.0140462)
    F: tuple[float, float] = (0.05779, 0.02962)
    G: tuple[float, float] = (0.10401, 0.02247)
    H: tuple[float, float] = (0.11557, 0.0246888)
    R: float = 0.0137543
    UPPER_EDGE: float = 0.045

    def area_ratio_inlet(self, dim: int) -> float:
        """Computes the ratio of the inlet area
        to the throat area.
        """
        if dim not in (2, 3):
            raise ValueError(f"Invalid dim. Got {dim}.")
        
        ratio = self.B[1] / self.R

        if dim == 2:
            return ratio
        else:
            return ratio**2
        
    @property
    def D_exit(self) -> float:
        return 2 * self.H[1]

    @property
    def D_throat(self) -> float:
        return 2 * self.R


@dataclass(frozen=True, slots=True)
class InjectorGeometry:
    X: float    # position
    IW: float   # width
    N: int      # count


@dataclass(frozen=True, slots=True)
class InjectorPlaneParameters:
    positions: Array
    tangents: Array
    normals: Array


@dataclass(frozen=True, slots=True)
class PressureRatios:
    NPR: float # nozzle pressure ratio
    SPR: float # secondary pressure ratio


class ObsData(NamedTuple):
    thrust_angle: Array
    target_angle: Array
    pressure_probes: Array | None


@dataclass(frozen=True, slots=True)
class TVCSpec:
    dim: int
    grid_resolutions: tuple[str, ...] = ("coarse", "fine")
    fixed_num_actuators: int | None = None
    min_num_actuators: int | None = None
    max_num_actuators: int | None = None
    ambient_pressure: float = 1e+5
    ambient_temperature: float = 300.0
    specific_gas_constant: float = 287.14
    specific_heat_ratio: float = 1.4
    nozzle_pressure_ratio: float = 4.6
    nozzle_geometry: NozzleGeometry = NozzleGeometry()
    injector_x: float = 0.789
    injector_width: float = 0.002032
    t_end: float = 1e-2

    @property
    def p0(self) -> float:
        return self.nozzle_pressure_ratio * self.ambient_pressure


@dataclass(frozen=True, slots=True)
class TVCEnvOptions:
    num_actuators: int
    secondary_pressure_ratio: float
    resolution: str
    ngpus: int
    is_pressure_probes: bool
    is_scale_observations: bool
    target_fn: TargetThrustAngleFn


@dataclass(frozen=True, slots=True)
class TVCRuntimeSetup:
    env_name: str
    env_dir: Path
    case_setup_dict: dict
    numerical_setup_dict: dict
    restart_file_path: Path


def build_tvc_env_options(
        *,
        env_config: dict,
        spec: TVCSpec,
        target_fns: dict[str, TargetThrustAngleFn],
        cls_name: str,
    ) -> TVCEnvOptions:

    num_actuators = env_config.get("num_actuators")
    if spec.fixed_num_actuators is not None:
        if num_actuators is not None:
            raise ValueError(
                f"{cls_name} requires {spec.fixed_num_actuators} actuators."
            )
        num_actuators = spec.fixed_num_actuators
    else:
        if num_actuators is None:
            raise ValueError("num_actuators must be provided.")

        min_num_actuators = spec.min_num_actuators
        max_num_actuators = spec.max_num_actuators
        if min_num_actuators is None or max_num_actuators is None:
            raise ValueError(
                f"{cls_name} must define either a fixed actuator count "
                "or both min_num_actuators and max_num_actuators."
            )

        if not (min_num_actuators <= num_actuators <= max_num_actuators):
            raise ValueError(
                f"num_actuators must be in "
                f"[{min_num_actuators}, {max_num_actuators}]. "
                f"Got {num_actuators}."
            )

    secondary_pressure_ratio = env_config.get("secondary_pressure_ratio", 0.7)
    if secondary_pressure_ratio < 0.7 or secondary_pressure_ratio > 0.9:
        raise ValueError(
            f"secondary_pressure_ratio must be >= 0.7 and <= 0.9."
            f"Got {secondary_pressure_ratio}."
        )

    resolution = env_config.get("resolution", "fine")
    if resolution not in spec.grid_resolutions:
        raise ValueError(
            f"Resolution {resolution} is not supported. "
            f"Please choose from {spec.grid_resolutions}."
        )

    ngpus = env_config.get("ngpus", 1)
    if not isinstance(ngpus, int):
        raise ValueError(f"ngpus must be of type int. Got {type(ngpus)}.")
    if ngpus < 1:
        raise ValueError(f"ngpus must be >= 1. Got {ngpus}.")

    is_pressure_probes = env_config.get("is_pressure_probes", False)
    if not isinstance(is_pressure_probes, bool):
        raise ValueError(
            "is_pressure_probes must be of type bool. "
            f"Got {type(is_pressure_probes)}"
        )

    is_scale_observations = env_config.get("is_scale_observations", True)
    if not isinstance(is_scale_observations, bool):
        raise ValueError(
            f"is_scale_observations needs to be of type bool. "
            f"Got {type(is_scale_observations)}."
        )

    target_key = env_config.get("target_fn", "sine")
    if target_key not in target_fns:
        raise ValueError(
            f"Unknown target_fn {target_key!r}. "
            f"Please choose from {tuple(target_fns)}."
        )

    return TVCEnvOptions(
        num_actuators=num_actuators,
        secondary_pressure_ratio=secondary_pressure_ratio,
        resolution=resolution,
        ngpus=ngpus,
        is_pressure_probes=is_pressure_probes,
        is_scale_observations=is_scale_observations,
        target_fn=target_fns[target_key],
    )


def build_tvc_runtime_setup(
        *,
        base_path: Path,
        dim: int,
        resolution: str,
        ngpus: int,
    ) -> TVCRuntimeSetup:

    env_name = f"Nozzle{dim}D_{resolution}"
    # env_dir = base_path / env_name
    env_dir = base_path

    case_setup_path = env_dir / "jxf_case_setup.json"
    numerical_setup_path = env_dir / "jxf_numerical_setup.json"
    restart_file_path = env_dir / "restart.h5"

    if not case_setup_path.exists():
        raise FileNotFoundError(f"Could not find case setup file {case_setup_path}.")

    if not numerical_setup_path.exists():
        raise FileNotFoundError(f"Could not find numerical setup file {numerical_setup_path}.")

    if not restart_file_path.exists():
        raise FileNotFoundError(f"Could not find restart file {restart_file_path}.")

    case_setup_dict = json.loads(case_setup_path.read_text())
    case_setup_dict["domain"]["decomposition"]["split_x"] = ngpus

    numerical_setup_dict = json.loads(numerical_setup_path.read_text())

    return TVCRuntimeSetup(
        env_name=env_name,
        env_dir=env_dir,
        case_setup_dict=case_setup_dict,
        numerical_setup_dict=numerical_setup_dict,
        restart_file_path=restart_file_path,
    )


def compute_thrust(
        primitives: Array, # shape (Np,Nx,Ny,Nz)
        p_infty: float,
        apertures_x: Array,
        cell_centers: tuple[Array, ...],
        cell_sizes: tuple[Array, ...],
    ) -> tuple[Array, Array, Array]:
    """Computes the thrust of the nozzle.

    F_x = mdot_e * u_e + (p_e - p_infty) * A_e
    F_y = mdot_e * v_e
    F_z = mdot_e * w_e
    """

    nozzle_geometry = NozzleGeometry()

    x, y, z = [xi.flatten() for xi in cell_centers]
    dx, _, _ = [dxi.flatten() for dxi in cell_sizes]

    DIM = 3 if len(z) > 1 else 2
    dx_min = jnp.min(dx)
    cell_face_area = dx_min if DIM == 2 else dx_min**2

    # interpolate primitives to cell face
    primitives_cf = jnp.concatenate([
        primitives[:,0:1], primitives, primitives[:,-1:]
    ], axis=1)
    primitives_cf = (primitives_cf[:,1:] + primitives_cf[:,:-1]) / 2

    x_cf = jnp.concatenate([x - dx/2, x[-1:] + dx[-1:]/2], axis=0)

    # x ids throat and exit
    xid_e = jnp.searchsorted(x_cf, nozzle_geometry.H[0]) - 1
    xid_t = jnp.searchsorted(x_cf, nozzle_geometry.F[0]) - 1

    _, Y, Z = jnp.meshgrid(x_cf, y, z, indexing="ij")
    if DIM == 2:
        mask = jnp.abs(Y[xid_e]) <= nozzle_geometry.H[1]
    else:
        mask = jnp.sqrt(Y[xid_e]**2 + Z[xid_e]**2) <= nozzle_geometry.H[1]

    # states at nozzle exit
    A_e = apertures_x[xid_e] * cell_face_area
    rho_e = primitives_cf[0, xid_e]
    vel_e = primitives_cf[1:4, xid_e]
    p_e = primitives_cf[4, xid_e]

    # states at throat
    A_t = apertures_x[xid_t] * cell_face_area
    rho_t = primitives_cf[0, xid_t]
    u_t = primitives_cf[1, xid_t]

    # mass flow
    mdot_e = rho_e * vel_e[0] * A_e * mask
    mdot_t = rho_t * u_t * A_t * mask

    # thrust 
    thrust = mdot_e * vel_e
    thrust = thrust.at[0].add( (p_e - p_infty) * A_e )
    thrust = jnp.sum(thrust * mask, axis=(-1,-2))

    # metrics
    mdot_e = jnp.sum(mdot_e, axis=(-1,-2))
    mdot_t = jnp.sum(mdot_t, axis=(-1,-2))

    return thrust, mdot_t, mdot_e


def _compute_injector_plane_params(
        injector_geometry: InjectorGeometry,
    ) -> InjectorPlaneParameters:
    """Computes position, normal and tangent of the injector.

    :param injector_geometry: _description_
    :type injector_geometry: InjectorGeometry
    :return: _description_
    :rtype: InjectorPlaneParameters
    """

    nozzle_geometry = NozzleGeometry()

    # compute vectors for base injector
    H = jnp.array(nozzle_geometry.H) # end point of convergent linear nozzle section
    E = jnp.array(nozzle_geometry.E) # start point of convergent linear nozzle section
    EH = H - E
    tangent = EH / jnp.linalg.norm(EH)
    x_position = E[0] + injector_geometry.X * (H[0] - E[0])
    y_position = (H[1] - E[1])/(H[0] - E[0]) * (x_position - E[0]) + E[1]

    t0 = jnp.array([tangent[0], tangent[1], 0.0])
    n0 = jnp.array([tangent[1], -tangent[0], 0.0])
    pos0 = jnp.array([x_position, y_position, 0.0])

    positions = []
    tangents = []
    normals = []

    theta = jnp.linspace(0, 2*jnp.pi, injector_geometry.N, endpoint=False)

    # rotate the vectors around the circumference of the nozzle 
    # to get the corresponding vectors of the remaining injectors
    for th in theta:
        R = jnp.array([
            [1, 0, 0],
            [0, jnp.cos(th), -jnp.sin(th)],
            [0, jnp.sin(th),  jnp.cos(th)],
        ])

        positions.append(R @ pos0)
        tangents.append(R @ t0)
        normals.append(R @ n0)

    positions = jnp.stack(positions)
    tangents = jnp.stack(tangents)
    normals = jnp.stack(normals)

    return InjectorPlaneParameters(
        positions, tangents, normals
    )


def _compute_injector_mask(
        mesh_grid: tuple[Array],
        IW: float,
        injector_params: InjectorPlaneParameters,
        dim: int,
        dx: float
    ) -> Array:

    position = injector_params.positions
    tangent = injector_params.tangents
    normal = injector_params.normals

    # project the distance vector R (mesh_grid - injector position)
    # on injector tangent plane to compute mask

    position = position[:, None]
    tangent = tangent[:, None]
    normal = normal[:, None]

    if dim == 2:
        X, Y = mesh_grid

        R = jnp.stack([
            X - position[0],
            Y - position[1]
        ], axis=0)

        s = jnp.sum(R*tangent[:2], axis=0)
        d = jnp.sum(R*normal[:2], axis=0)

        mask_injector = (
            (jnp.abs(s) <= IW / 2) &
            (jnp.abs(d) <= dx)
        )

    elif dim == 3:
        X, Y, Z = mesh_grid

        R = jnp.stack([
            X - position[0],
            Y - position[1],
            Z - position[2]
        ], axis=0)

        # compute circumferential tangent
        t_theta = jnp.cross(normal, tangent, axis=0)
        t_theta = t_theta / jnp.linalg.norm(t_theta, keepdims=True)

        # projections
        s_x = jnp.sum(R*tangent, axis=0)
        s_theta = jnp.sum(R*t_theta, axis=0)
        d = jnp.sum(R*normal, axis=0)

        R_inj = IW / 2

        mask_injector = (
            (s_x**2 + s_theta**2 <= R_inj**2) &
            (jnp.abs(d) <= 5*dx) # NOTE large safety offset in normal direction given that nozzle exit surface is curved around circumference
        )

    else:
        raise ValueError

    return mask_injector


def _compute_choked_state(
        p_total_injector: float,
        rho_total_injector: float,
        specific_heat_ratio: float,
    ) -> tuple[float, float, float, float]:
    p_choked = p_total_injector * pressure_ratio_isentropic(1.0, specific_heat_ratio)
    rho_choked = rho_total_injector * density_ratio_isentropic(1.0, specific_heat_ratio)
    u_choked = speed_of_sound(p_choked, rho_choked, specific_heat_ratio)
    E_choked = total_energy(p_choked, rho_choked, u_choked, specific_heat_ratio)
    return p_choked, rho_choked, u_choked, E_choked 


def _compute_unchoked_state(
        p_local: float,
        rho_total_injector: float,
        p_ratio: float,
        specific_heat_ratio: float
    ) -> tuple[float, float, float]:
    # NOTE if p_ratio_unchoked <= 1.0 -> no injection. We clip to prevent negative sqrt.
    p_ratio_unchoked = jnp.clip(p_ratio, 0.0, 1.0)
    M_unchoked = mach_number_from_pressure_ratio_isentropic(p_ratio_unchoked, specific_heat_ratio)
    rho_unchoked = rho_total_injector * density_ratio_isentropic(M_unchoked, specific_heat_ratio)
    u_unchoked = M_unchoked * speed_of_sound(p_local, rho_unchoked, specific_heat_ratio)
    E_unchoked = total_energy(p_local, rho_unchoked, u_unchoked, specific_heat_ratio) 
    return rho_unchoked, u_unchoked, E_unchoked 


def initialize_injector_flux_fn(
        injector_geometry: InjectorGeometry,
        pressure_ratios: PressureRatios,
        p_infty: float,
        T_infty: float,
        specific_heat_ratio: float,
        specific_gas_constant: float,
        sim_manager: SimulationManager,
    ) -> Callable[
            [Array, Array, Array, Array, Array],
            tuple[Array, Array, Array]
        ]:

    domain_information = sim_manager.domain_information
    dim = domain_information.dim
    dx = domain_information.smallest_cell_size

    IW = injector_geometry.IW
    NPR = pressure_ratios.NPR
    SPR = pressure_ratios.SPR

    # total pressure in reservoir for injector
    P0 = p_infty * NPR * SPR

    num_injectors = injector_geometry.N
    injector_params = _compute_injector_plane_params(injector_geometry)

    def compute_interface_flux(
            primitives: Array,
            interface_length: Array,
            normal: Array,
            mesh_grid: Array,
            actuator: Array
        ) -> tuple[Array, Array, Array]:
        """Computes the interface flux for each injector.
        """
        
        if len(actuator) != num_injectors:
            raise ValueError("Number of actions unequal to injector count")

        # local nozzle pressure
        p_local = primitives[-1]

        # clipping actions
        actuator = jnp.clip(actuator, 0.0, 1.0)

        mass_flux = 0.0
        momentum_flux = p_local * normal * interface_length
        energy_flux = 0.0

        for i in range(num_injectors):

            actuator_i = actuator[i]

            # actuator [0.0, 1.0] adjusts total pressure
            p_total_injector = p_local + actuator_i * (P0 - p_local)
            rho_total_injector = density_from_pressure_temperature(
                p=p_total_injector,
                T=T_infty,
                R=specific_gas_constant,
            ) 

            # we assume isentropic expansion to either M=1 (choked)
            # or to local pressure to compute injector state

            # choked injector state
            p_choked, rho_choked, u_choked, E_choked = _compute_choked_state(
                p_total_injector=p_total_injector,
                rho_total_injector=rho_total_injector,
                specific_heat_ratio=specific_heat_ratio,
            )
            
            # unchoked injector state
            p_ratio = p_local / p_total_injector
            rho_unchoked, u_unchoked, E_unchoked = _compute_unchoked_state(
                p_local=p_local,
                rho_total_injector=rho_total_injector,
                p_ratio=p_ratio,
                specific_heat_ratio=specific_heat_ratio
            )

            # checking if injector is choked
            p_ratio_choked = pressure_ratio_isentropic(1.0, specific_heat_ratio)
            mask_choked = p_ratio < p_ratio_choked

            # masking choked vs. unchoked
            rho_injector = jnp.where(mask_choked, rho_choked, rho_unchoked)
            p_injector = jnp.where(mask_choked, p_choked, p_local)
            u_injector = jnp.where(mask_choked, u_choked, u_unchoked)
            E_injector = jnp.where(mask_choked, E_choked, E_unchoked)

            mask_injector = _compute_injector_mask(
                mesh_grid, IW,
                InjectorPlaneParameters(
                    injector_params.positions[i],
                    injector_params.tangents[i],
                    injector_params.normals[i]
                ),
                dim, dx
            )

            u_injector_vec = u_injector * injector_params.normals[i][:,None]
            u_injector_n = jnp.sum(u_injector_vec * normal, axis=0)

            mass_flux += rho_injector * u_injector_n * interface_length * mask_injector
            momentum_flux_i = (
                rho_injector * u_injector_n * u_injector_vec + p_injector * normal
            ) * interface_length
            momentum_flux = momentum_flux * (1 - mask_injector) + momentum_flux_i * mask_injector
            energy_flux += (E_injector + p_injector) * u_injector_n * interface_length * mask_injector
            
        return mass_flux, momentum_flux, energy_flux

    return compute_interface_flux


def plot_flowfield_3d(
        primitives: ndarray,
        levelset: ndarray,
        cell_centers: tuple[ndarray, ...],
        cell_sizes: tuple[ndarray, ...],
        nozzle_pressure: float,
        injector_geometry: InjectorGeometry,
        specific_heat_ratio: float,
    ) -> pv.Plotter:


    injector_params = _compute_injector_plane_params(injector_geometry)

    mesh_grid = np.meshgrid(*cell_centers, indexing="ij")
    mesh_grid = np.stack(mesh_grid, axis=0)
    IW = injector_geometry.IW
    min_dx = np.min(cell_sizes[0])

    mask_injector_list = []

    for i in range(injector_geometry.N):
        mask_injector = _compute_injector_mask(
            mesh_grid.reshape(3,-1), IW,
            InjectorPlaneParameters(
                injector_params.positions[i],
                injector_params.tangents[i],
                injector_params.normals[i]
            ),
            3, min_dx
        )
        mask_injector_list.append(mask_injector.reshape(mesh_grid[0].shape))

    # compute fields
    density = primitives[0]
    velocity = np.linalg.norm(primitives[1:4],axis=0,ord=2)
    pressure = primitives[-1]
    speed_of_sound = np.sqrt(specific_heat_ratio * pressure / density)
    mach_number = velocity/speed_of_sound
    schlieren = np.linalg.norm(np.gradient(density),axis=0,ord=2)

    min_dx = np.min(cell_sizes[0])

    # Build stretched grid
    x_centers, y_centers, z_centers = cell_centers
    dx, dy, dz = cell_sizes

    # Compute cell edges from centers + sizes
    x_edges = np.concatenate(([x_centers[0] - dx[0] / 2], x_centers + dx / 2))
    y_edges = np.concatenate(([y_centers[0] - dy[0] / 2], y_centers + dy / 2))
    z_edges = np.concatenate(([z_centers[0] - dz[0] / 2], z_centers + dz / 2))

    # Create rectilinear grid (supports stretched mesh)
    grid_levelset = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)

    # Attach data
    total_mask = np.maximum.reduce(mask_injector_list).astype(float)
    grid_levelset.cell_data["levelset"] = levelset.ravel(order="F") / min_dx
    grid_levelset.cell_data["pressure"] = pressure.ravel(order="F") / nozzle_pressure
    grid_levelset.cell_data["total_mask"] = total_mask.ravel(order="F")

    grid.cell_data["levelset"] = levelset.ravel(order="F")/min_dx
    grid.cell_data["schlieren"] = schlieren.ravel(order="F")
    grid.cell_data["mach_number"] = mach_number.ravel(order="F")

    # Ensure we are working with point data
    grid_levelset = grid_levelset.cell_data_to_point_data()
    grid = grid.cell_data_to_point_data()
    grid, grid_levelset = clip_grids(grid, grid_levelset)

    # Plot
    plotter = plot_slice(grid_levelset, grid)

    return plotter


def clip_grids(grid, grid_levelset):

    # --- compute shared geometry once ---
    points = grid.points  # use one grid (same topology assumed)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(y**2 + z**2)

    # --- build masks ---
    nozzle_x_limit = NozzleGeometry().H[0] - 5e-4

    mask_levelset = (
        (r < 0.044) &
        (x < nozzle_x_limit)
    )

    mask_grid = (
        (r < 0.044) &
        (x < 0.25) &
        (grid.point_data["levelset"] > 2)
    )

    # --- extract in one pass ---
    grid_levelset_clipped = grid_levelset.extract_points(
        mask_levelset,
        adjacent_cells=True
    )

    grid_clipped = grid.extract_points(
        mask_grid,
        adjacent_cells=True
    )

    return grid_clipped, grid_levelset_clipped

def plot_slice(grid_levelset, grid) -> pv.Plotter:

    plotter = pv.Plotter(off_screen=True, window_size=(3000, 2200))

    # CMAP = "Spectral_r"
    CMAP = "coolwarm"

    # nozzle
    contour = grid_levelset.contour(isosurfaces=[0.0], scalars="levelset")
    contour = contour.triangulate().subdivide(2, subfilter="linear")
    contour = contour.sample(grid_levelset)

    contour_visible = contour.threshold(
        value=0.5,
        scalars="total_mask",
        invert=True,
    ).extract_surface(algorithm="dataset_surface")

    contour_masked = contour.threshold(
        value=0.5,
        scalars="total_mask",
        invert=False,
    ).extract_surface(algorithm="dataset_surface")

    contour_masked = contour.threshold(
        value=0.5,
        scalars="total_mask",
        invert=False,
    ).extract_surface(algorithm="dataset_surface")

    contour_masked = contour_masked.compute_normals(
        cell_normals=False,
        point_normals=True,
    )
    contour_masked = contour_masked.warp_by_vector("Normals", factor=2e-4)

    masked_outline = contour_masked.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    plotter.add_mesh(
        contour_visible,
        scalars="pressure",
        cmap=CMAP,
        opacity=1.0,
        smooth_shading=True,
        show_scalar_bar=False,
        clim=[0.0, 1.0],
    )

    plotter.add_mesh(
        contour_masked,
        color="white",
        opacity=1.0,
        smooth_shading=False,
        show_scalar_bar=False,
        lighting=False,
    )

    plotter.add_mesh(
        masked_outline,
        color="black",
        line_width=5.0,
        lighting=False,
    )


    # mach number
    offset = 0.05
    slice_xy = grid.slice(
        normal=(0, 0, 1),
        origin=(0.0, 0.0, 0.0)
    )
    slice_xy = slice_xy.threshold(
        value=0.0,
        scalars="levelset",
    )
    slice_xy = slice_xy.translate((0.0, 0.0, -offset), inplace=False)
    plotter.add_mesh(
        slice_xy,
        scalars="mach_number",
        cmap=CMAP,
        clim=[0.0, 3.0],
        show_scalar_bar=False,
        lighting=True,
        ambient=0.1,
        # diffuse=1.0,
        # specular=0.1,
    )

    slice_xz = grid.slice(
        normal=(0, 1, 0),
        origin=(0.0, 0.0, 0.0)
    )
    slice_xz = slice_xz.threshold(
        value=0.0,
        scalars="levelset",
    )
    slice_xz = slice_xz.translate((0.0, -offset, 0.0), inplace=False)
    plotter.add_mesh(
        slice_xz,
        scalars="mach_number",
        cmap=CMAP,
        clim=[0.0, 3.0],
        show_scalar_bar=False,
        # lighting=True,
        # ambient=0.05,
        # diffuse=0.5,
        # specular=0.1,
    )

    # schlieren
    low = 0.25
    high = 1.5
    n = 20

    values = np.linspace(low, high, n)
    contours = grid.contour(
        isosurfaces=values,
        scalars="schlieren"
    )
    plotter.add_mesh(
        contours,
        color="darkgray",
        opacity=0.3,
        smooth_shading=True,
        lighting=True,
        # ambient=0.2,
        diffuse=1.0,
        specular=1.0,
        specular_power=100,
    )

    plotter.add_light(pv.Light(light_type='headlight', intensity=0.3))
    # plotter.add_axes()
    # plotter.show_bounds()
    # plotter.show_grid()
    plotter.camera_position = (
            (0.25,0.1,0.15),
            (0.12,0.0,0.0),
            (0,1,0),
    )
    # plotter.enable_eye_dome_lighting()
    # plotter.set_background("black")

    return plotter