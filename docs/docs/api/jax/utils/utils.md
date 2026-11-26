---
sidebar_label: utils
title: hydrogym.jax.utils.utils
---

#### dealias\_mask\_2\_3

```python
def dealias_mask_2_3(Nx: int, Ny: int)
```

Standard 2/3 rule in x,y for modes after fftfreq ordering.

#### cheb\_D\_matrices

```python
def cheb_D_matrices(N: int, Lz: float)
```

Chebyshev-Gauss-Lobatto points on [-1,1], and D1/D2 scaled to physical z in [0,Lz] or [-Lz/2,Lz/2]
We keep points on [-1,1] and scale derivatives by (2/Lz).

#### compute\_velocity\_fft

```python
def compute_velocity_fft(omega_hat, kx, ky)
```

Computing the fourier velocity components (u_hat, v_hat) from the stream function (phi_hat)
(Yin, Z. 2004)

**Arguments**:

- `omega_hat` - the Fourier transform of the vorticity
- `kx` - wavenumbers x
- `ky` - wavenumbers y
  

**Returns**:

  Tuple `(u_hat, v_hat)`.

#### dealiasing

```python
def dealiasing(advection_term, m=None)
```

2/3 dealiasing rule for a 2D field in `rfftn` layout: the Fourier modes with
`|k_x| > n // 3` or `k_y > m // 3` are set to 0 (the same cut-off as
`dealias_mask_2_3`).
Reference: S. A. Orszag, J. Atmos. Sci. 28, 1074 (1971) (the 2/3 rule).

**Arguments**:

- `advection_term` - field in Fourier space, shape `(n, m // 2 + 1)`.
- `m` - length of the real-space axis that was transformed with `rfft`
  (default `2 * (advection_term.shape[1] - 1)`, i.e. an even grid).
  

**Returns**:

  The truncated field (same shape and dtype).

#### compute\_energy\_mode

```python
def compute_energy_mode(uhat, vhat, kx, ky, n, m)
```

Compute the energy of a specific mode and wavenumber.

**Arguments**:

  uhat, vhat: fft velocity components
- `kx` - wavenumber x
- `ky` - wavenumber y
  n, m: grid size

#### compute\_velocity\_mode

```python
def compute_velocity_mode(uhat, vhat, kx, ky, n, m)
```

Compute the velocity of a specific mode and wavenumber.

**Arguments**:

  uhat, vhat: fft velocity components
- `kx` - wavenumber x
- `ky` - wavenumber y
  n, m: grid size

#### compute\_real\_velocity\_point

```python
def compute_real_velocity_point(uhat, vhat, x_idx, y_idx)
```

Compute the velocity of a specific point in real space.

**Arguments**:

  uhat, vhat: fft velocity components
- `x_idx` - grid index of the point in x
- `y_idx` - grid index of the point in y

#### compute\_energy\_dissipation

```python
def compute_energy_dissipation(omega_hat, kx, ky, nu, n)
```

Computes the energy dissipation of the system given the fft vorticity field.
The instantaneous energy dissipation rate can be estimated by:
Ɛ(x,t) = 2v&lt;(S_ij S_ij)> [2]
where S_ij denotes the fluctuation strain-rate tensor and v denotes the kinematic viscosity [1,2].
[1] Pope, 2000
[2] Buaria et. al, eq 1.1 in doi: 10.1098/rsta.2021.0088

**Arguments**:

- `omega_hat` - fft vorticity
- `kx` - wavenumber x
- `ky` - wavenumber y
- `nu` - kinematic viscosity
- `n` - grid length (not used)

#### compute\_tke

```python
def compute_tke(omega_hat, kx, ky, n)
```

Computes the TKE of the system given the fft vorticity field.

**Arguments**:

- `omega_hat` - fft vorticity
- `kx` - wavenumber x
- `ky` - wavenumber y
- `n` - grid length

#### compute\_reward

```python
def compute_reward(omega_hat, kx, ky, nu, n, actions)
```

Computes the energy dissipation of the system given the fft vorticity field.

**Arguments**:

- `omega_hat` - fft vorticity
- `kx` - wavenumber x
- `ky` - wavenumber y
- `nu` - kinematic viscosity
- `n` - grid length

#### compute\_divergence

```python
def compute_divergence(omega_hat, kx, ky)
```

Computes the divergence of the system given the fft vorticity field.

**Arguments**:

- `omega_hat` - fft vorticity
- `kx` - wavenumber x
- `ky` - wavenumber y

