
from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.utils.utils import *
from hydrogym.jax.flow import FlowConfig
from hydrogym.jax.equation import * 
from hydrogym.jax.solvers.base import *

class PseudoSpectralNavierStokes3D(SplitEquation):
    
    def __init__(self, flow: "FlowConfig"):
        self.flow = flow
        self.Nx = flow.Nx
        self.Ny = flow.Ny
        self.Nz = flow.Nz
        self.Lx = flow.Lx
        self.Ly = flow.Ly
        self.Lz = flow.Lz
        self.nu = flow.nu
 
        kx = jnp.fft.fftfreq(self.Nx, d=self.Lx / self.Nx) * 2.0 * jnp.pi
        ky = jnp.fft.fftfreq(self.Ny, d=self.Ly / self.Ny) * 2.0 * jnp.pi

        self.kx = kx
        self.ky = ky
        self.ikx = (1j * kx)[:, None, None]
        self.iky = (1j * ky)[None, :, None]
        self.k2 = (kx[:, None] ** 2 + ky[None, :] ** 2)

        self.z, self.Dz, self.Dzz = cheb_D_matrices(self.Nz, self.Lz)
        self.dealias = dealias_mask_2_3(self.Nx, self.Ny)[:, :, None]

    def fft_xy(self, f):
        return jnp.fft.fftn(f, axes=(0, 1))

    def ifft_xy(self, f_hat):
        return jnp.fft.ifftn(f_hat, axes=(0, 1)).real

    def to_spectral(self, state: "VelocityState") -> "VelocityState":
        return VelocityState(
            self.fft_xy(state.u),
            self.fft_xy(state.v),
            self.fft_xy(state.w),
        )

    def to_physical(self, state_hat: "VelocityState") -> "VelocityState":
        
        return VelocityState(
            self.ifft_xy(state_hat.u),
            self.ifft_xy(state_hat.v),
            self.ifft_xy(state_hat.w),
        )

    def dx_hat(self, f_hat):
        return self.ikx * f_hat

    def dy_hat(self, f_hat):
        return self.iky * f_hat

    def dz_phys(self, f_phys):
        return jnp.einsum("ij,xyj->xyi", self.Dz, f_phys)

    def dzz_phys(self, f_phys):
        return jnp.einsum("ij,xyj->xyi", self.Dzz, f_phys)

    def enforce_noslip(self, u, v, w):
        u = u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        v = v.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        w = w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        return u, v, w

    def forcing_term(self):
        forcing_func = getattr(self.flow, "forcing_function", None)
        if forcing_func is None:
            zeros = jnp.zeros((self.Nx, self.Ny, self.Nz))
            return zeros, zeros, zeros

        x, y, z = self.flow.load_mesh("name")
        fx, fy, fz = forcing_func(x=x, y=y, z=z)
        return fx, fy, fz

    def control_term(self):
        # Treated as a field tuple, not a callable
        control_field = getattr(self.flow, "control_function", None)
        if control_field is None:
            zeros = jnp.zeros((self.Nx, self.Ny, self.Nz))
            return zeros, zeros, zeros

        cx, cy, cz = control_field
        return cx, cy, cz

    def nonlinear_terms(self, state_hat: "VelocityState") -> "VelocityState":
        state = self.to_physical(state_hat)
        u, v, w = self.enforce_noslip(state.u, state.v, state.w)

        u_hat = self.fft_xy(u)
        v_hat = self.fft_xy(v)
        w_hat = self.fft_xy(w)

        du_dx = self.ifft_xy(self.dx_hat(u_hat))
        du_dy = self.ifft_xy(self.dy_hat(u_hat))
        du_dz = self.dz_phys(u)

        dv_dx = self.ifft_xy(self.dx_hat(v_hat))
        dv_dy = self.ifft_xy(self.dy_hat(v_hat))
        dv_dz = self.dz_phys(v)

        dw_dx = self.ifft_xy(self.dx_hat(w_hat))
        dw_dy = self.ifft_xy(self.dy_hat(w_hat))
        dw_dz = self.dz_phys(w)

        Nu = -(u * du_dx + v * du_dy + w * du_dz)
        Nv = -(u * dv_dx + v * dv_dy + w * dv_dz)
        Nw = -(u * dw_dx + v * dw_dy + w * dw_dz)

        fx, fy, fz = self.forcing_term()
        cx, cy, cz = self.control_term()

        Nu = Nu + fx + cx
        Nv = Nv + fy + cy
        Nw = Nw + fz + cz

        Nu_hat = self.fft_xy(Nu) * self.dealias
        Nv_hat = self.fft_xy(Nv) * self.dealias
        Nw_hat = self.fft_xy(Nw) * self.dealias

        return VelocityState(Nu_hat, Nv_hat, Nw_hat)

    def linear_terms(self, state_hat: "VelocityState") -> "VelocityState":
        state = self.to_physical(state_hat)
        u, v, w = self.enforce_noslip(state.u, state.v, state.w)

        uzz_hat = self.fft_xy(self.dzz_phys(u))
        vzz_hat = self.fft_xy(self.dzz_phys(v))
        wzz_hat = self.fft_xy(self.dzz_phys(w))

        k2 = self.k2[:, :, None]

        Lu = self.nu * (-k2 * state_hat.u + uzz_hat)
        Lv = self.nu * (-k2 * state_hat.v + vzz_hat)
        Lw = self.nu * (-k2 * state_hat.w + wzz_hat)

        return VelocityState(Lu, Lv, Lw)

    def implicit_timestep(self, state_hat: "VelocityState", time_step: float) -> "VelocityState":
        state = self.to_physical(state_hat)

        uzz_hat = self.fft_xy(self.dzz_phys(state.u))
        vzz_hat = self.fft_xy(self.dzz_phys(state.v))
        wzz_hat = self.fft_xy(self.dzz_phys(state.w))

        k2 = self.k2[:, :, None]
        denom = 1.0 + time_step * self.nu * k2

        u_new = (state_hat.u + time_step * self.nu * uzz_hat) / denom
        v_new = (state_hat.v + time_step * self.nu * vzz_hat) / denom
        w_new = (state_hat.w + time_step * self.nu * wzz_hat) / denom

        return VelocityState(u_new, v_new, w_new)

    def project(self, state_hat: "VelocityState", dt: float) -> "VelocityState":
        u_hat, v_hat, w_hat = state_hat.u, state_hat.v, state_hat.w

        du_dx = self.ifft_xy(self.dx_hat(u_hat))
        dv_dy = self.ifft_xy(self.dy_hat(v_hat))
        w = self.ifft_xy(w_hat)
        dw_dz = self.dz_phys(w)

        div_phys = du_dx + dv_dy + dw_dz
        rhs_hat = self.fft_xy(div_phys) / dt

        Nz = self.Nz
        I = jnp.eye(Nz)
        Dzz = self.Dzz
        Dz = self.Dz

        Nm = self.Nx * self.Ny
        k2_flat = self.k2.reshape(-1)
        rhs_flat = rhs_hat.reshape(Nm, Nz)

        A = Dzz[None, :, :] - k2_flat[:, None, None] * I[None, :, :]
        A = A.at[:, 0, :].set(Dz[0, :][None, :])
        A = A.at[:, -1, :].set(Dz[-1, :][None, :])

        rhs = rhs_flat
        rhs = rhs.at[:, 0].set(0.0)
        rhs = rhs.at[:, -1].set(0.0)

        mid = Nz // 2
        A0 = A[0].at[mid, :].set(0.0).at[mid, mid].set(1.0)
        b0 = rhs[0].at[mid].set(0.0)
        A = A.at[0].set(A0)
        rhs = rhs.at[0].set(b0)

        p_flat = jax.vmap(jnp.linalg.solve, in_axes=(0, 0))(A, rhs)
        p_hat = p_flat.reshape(self.Nx, self.Ny, Nz)

        u_hat_new = u_hat - dt * self.ikx * p_hat
        v_hat_new = v_hat - dt * self.iky * p_hat

        p = self.ifft_xy(p_hat)
        dp_dz = self.dz_phys(p)
        dp_dz_hat = self.fft_xy(dp_dz)
        w_hat_new = w_hat - dt * dp_dz_hat

        proj_state = VelocityState(u_hat_new, v_hat_new, w_hat_new)
        proj_phys = self.to_physical(proj_state)
        u, v, w = self.enforce_noslip(proj_phys.u, proj_phys.v, proj_phys.w)

        return self.to_spectral(VelocityState(u, v, w))

    def rhs(self, state_hat: "VelocityState") -> "VelocityState":
        N = self.nonlinear_terms(state_hat)
        L = self.linear_terms(state_hat)
        return VelocityState(
            N.u + L.u,
            N.v + L.v,
            N.w + L.w,
        )