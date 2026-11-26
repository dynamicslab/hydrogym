import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from jax import lax
from matplotlib.animation import PillowWriter


def dealias_mask_2_3(Nx: int, Ny: int):
    """
    Standard 2/3 rule in x,y for modes after fftfreq ordering.
    """
    kx = jnp.fft.fftfreq(Nx) * Nx  # integer-like
    ky = jnp.fft.fftfreq(Ny) * Ny
    kx_cut = Nx // 3
    ky_cut = Ny // 3
    mx = (jnp.abs(kx) <= kx_cut)[:, None]
    my = (jnp.abs(ky) <= ky_cut)[None, :]
    mask = (mx & my).astype(jnp.float32)  # (Nx,Ny)
    return mask


def cheb_D_matrices(N: int, Lz: float):
    """
    Chebyshev-Gauss-Lobatto points on [-1,1], and D1/D2 scaled to physical z in [0,Lz] or [-Lz/2,Lz/2]
    We keep points on [-1,1] and scale derivatives by (2/Lz).
    """
    k = jnp.arange(N)
    x = jnp.cos(jnp.pi * k / (N - 1))  # on [1,-1]
    x = x[::-1]  # make ascending [-1,1]

    c = jnp.ones(N)
    c = c.at[0].set(2.0)
    c = c.at[-1].set(2.0)
    c = c * ((-1.0) ** jnp.arange(N))

    X = x[:, None]
    dX = X - X.T

    # Off-diagonal entries
    D = (c[:, None] / c[None, :]) / (dX + jnp.eye(N))  # add eye to avoid /0; will fix diagonal next
    D = D - jnp.diag(jnp.sum(D, axis=1))  # set diagonal so rows sum to 0

    # Scale to physical z: z = (Lz/2)*x => d/dz = (2/Lz)*d/dx
    scale1 = 2.0 / Lz
    D1 = scale1 * D
    D2 = (scale1**2) * (D @ D)

    # Physical z points: map [-1,1] -> [-Lz/2, Lz/2]
    z = (Lz / 2.0) * x
    return z, D1, D2


def compute_velocity_fft(omega_hat, kx, ky):
    """
    Computing the fourier velocity components (u_hat, v_hat) from the stream function (phi_hat)
    (Yin, Z. 2004)

    Args:
        omega_hat: the Fourier transform of the vorticity
        kx: wavenumbers x
        ky: wavenumbers y

    Returns:
        Tuple ``(u_hat, v_hat)``.
    """
    double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(kx) ** 2 + abs(ky) ** 2)
    double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

    psi_hat = -1 * omega_hat / double_derivative
    vxhat = (2 * jnp.pi * 1j) * ky * psi_hat  # Get u,v from phi
    vyhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat
    return vxhat, vyhat


def dealiasing(advection_term, m=None):
    """
    2/3 dealiasing rule for a 2D field in ``rfftn`` layout: the Fourier modes with
    ``|k_x| > n // 3`` or ``k_y > m // 3`` are set to 0 (the same cut-off as
    ``dealias_mask_2_3``).
    Reference: S. A. Orszag, J. Atmos. Sci. 28, 1074 (1971) (the 2/3 rule).

    Args:
        advection_term: field in Fourier space, shape ``(n, m // 2 + 1)``.
        m: length of the real-space axis that was transformed with ``rfft``
            (default ``2 * (advection_term.shape[1] - 1)``, i.e. an even grid).

    Returns:
        The truncated field (same shape and dtype).
    """
    n, mh = advection_term.shape[0], advection_term.shape[1]
    m = 2 * (mh - 1) if m is None else int(m)
    i = jnp.arange(n)
    kx = jnp.abs(jnp.where(i < (n + 1) // 2, i, i - n))  # integer |k_x| in fftfreq order
    ky = jnp.arange(mh)
    keep = (kx <= n // 3)[:, None] & (ky <= m // 3)[None, :]
    return jnp.where(keep, advection_term, jnp.zeros_like(advection_term))


def compute_energy_mode(uhat, vhat, kx, ky, n, m):
    """
    Compute the energy of a specific mode and wavenumber.

    Args:
        uhat, vhat: fft velocity components
        kx: wavenumber x
        ky: wavenumber y
        n, m: grid size
    """

    # Compute indices of wavenumber
    kx_idx = kx % n
    ky_idx = ky % m
    energy = 0.5 * (jnp.abs(uhat[kx_idx, ky_idx]) ** 2 + jnp.abs(vhat[kx_idx, ky_idx]) ** 2) / ((n * m) ** 2)
    return energy


def compute_velocity_mode(uhat, vhat, kx, ky, n, m):
    """
    Compute the velocity of a specific mode and wavenumber.

    Args:
        uhat, vhat: fft velocity components
        kx: wavenumber x
        ky: wavenumber y
        n, m: grid size
    """

    # Compute indices of wavenumber
    kx_idx = kx % n
    ky_idx = ky % m
    velocity_mag = jnp.sqrt(
        (jnp.abs(uhat[kx_idx, ky_idx]) ** 2 + jnp.abs(vhat[kx_idx, ky_idx]) ** 2) / jnp.float64(n * m) ** 2
    )
    return velocity_mag


def compute_real_velocity_point(uhat, vhat, x_idx, y_idx):
    """
    Compute the velocity of a specific point in real space.

    Args:
        uhat, vhat: fft velocity components
        x_idx: grid index of the point in x
        y_idx: grid index of the point in y
    """
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    velocity_mag = jnp.sqrt((jnp.abs(ureal[x_idx, y_idx]) ** 2 + jnp.abs(vreal[x_idx, y_idx]) ** 2))
    return velocity_mag


def compute_energy_dissipation(omega_hat, kx, ky, nu, n):
    """
    Computes the energy dissipation of the system given the fft vorticity field.
    The instantaneous energy dissipation rate can be estimated by:
        Ɛ(x,t) = 2v<(S_ij S_ij)> [2]
    where S_ij denotes the fluctuation strain-rate tensor and v denotes the kinematic viscosity [1,2].
    [1] Pope, 2000
    [2] Buaria et. al, eq 1.1 in doi: 10.1098/rsta.2021.0088

    Args:
        omega_hat: fft vorticity
        kx: wavenumber x
        ky: wavenumber y
        nu: kinematic viscosity
        n: grid length (not used)
    """

    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    du_dy = jnp.gradient(ureal, axis=1)
    dv_dx = jnp.gradient(vreal, axis=0)
    du_dx = jnp.gradient(ureal, axis=0)
    dv_dy = jnp.gradient(vreal, axis=1)
    avg_epsilon = 2 * nu * ((du_dx) ** 2 + (dv_dy) ** 2 + (du_dy + dv_dx) ** 2)
    epsilon = jnp.sum(avg_epsilon) / (4 * jnp.pi**2)

    return epsilon


def compute_tke(omega_hat, kx, ky, n):
    """
    Computes the TKE of the system given the fft vorticity field.

    Args:
        omega_hat: fft vorticity
        kx: wavenumber x
        ky: wavenumber y
        n: grid length
    """

    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    u = jnp.fft.irfftn(uhat)
    v = jnp.fft.irfftn(vhat)
    U = jnp.mean(u)
    V = jnp.mean(v)
    avg_tke = 0.5 * (jnp.abs(u - U) ** 2 + jnp.abs(v - V) ** 2) * (1 / n**2)
    tke = jnp.sum(avg_tke)

    return tke


def compute_reward(omega_hat, kx, ky, nu, n, actions):
    """
    Computes the energy dissipation of the system given the fft vorticity field.

    Args:
        omega_hat: fft vorticity
        kx: wavenumber x
        ky: wavenumber y
        nu: kinematic viscosity
        n: grid length
    """
    tke = compute_tke(omega_hat, kx, ky, n)
    actions_cost = actions[0] + actions[1] + actions[2] + actions[3]
    return tke - actions_cost


def compute_divergence(omega_hat, kx, ky):
    """
    Computes the divergence of the system given the fft vorticity field.

    Args:
        omega_hat: fft vorticity
        kx: wavenumber x
        ky: wavenumber y
    """

    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    du_dx = jnp.gradient(ureal, axis=0)
    dv_dy = jnp.gradient(vreal, axis=1)

    return du_dx + dv_dy
