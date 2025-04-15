import tree_math
from typing import Callable, Iterable, Tuple
import jax.numpy as jnp 
import numpy as np
import jax
from jax import lax

from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.utils.utils import *
from hydrogym.jax.flow import FlowConfig 

_alpha_RK4 = [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
_beta_RK4 = [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
_gammas_RK4 = [0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

class PseudoSpectralNavierStokes2D:
  """ 
  Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver. We transform the 2D Navier-Stokes equation to a vorticity equation:
    ∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
    ω = - ∇²φ ; 
  and solve in Fourier space
  
  """

  def __init__(self, flow: FlowConfig): 
      self.flow = flow 
      self.grid = flow.load_fft_mesh()
      self.real_grid = flow.load_mesh('name')
      self.kx, self.ky = self.grid
      self.x, self.y = self.real_grid
          
  def linear_terms(self, omega_hat):
    """Computes the linear (viscous) term of the vorticity equation
    """
    return self.flow.nu *  (2j * jnp.pi)**2 * (self.kx**2 + self.ky**2) * omega_hat
   
  def implicit_timestep(self,omega_hat, time_step):
    """
    Function that computes an implicit euler timestep,
      y_n+1 = y_n / (1-∇tλ). 
    
    """
    double_derivative = (2j * jnp.pi)**2 * (self.kx**2 + self.ky**2)
    return 1 / (1 - time_step * self.flow.nu * double_derivative) * omega_hat
    
  def nonlinear_terms(self, omega_hat):
    """Computes the explicit (nonlinear) terms in the vorticity equation. 
    Uses the stream function to compute velocity components in Fourier space.

    Args:
        omega_hat: fft of vorticity

    Returns:
        terms: Nonlinear terms of the equation.
    """
    
    kx, ky = self.kx, self.ky
    
    double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(self.kx)**2 + abs(ky)**2)
    double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

    psi_hat = -1 * omega_hat / double_derivative 
    uhat = (2 * jnp.pi * 1j) * ky * psi_hat # Get u,v from phi 
    vhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat
  
    u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)

    grad_x_hat = 2j * jnp.pi * self.kx * omega_hat
    grad_y_hat = 2j * jnp.pi * self.ky * omega_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * u + grad_y * v)
    advection_hat = jnp.fft.rfftn(advection)
    
    forcing_hat = self.forcing_term()
    control_hat = self.control_term()
    advection_hat = dealiasing(advection_hat) # 2/3 dealiasing rule

    terms = advection_hat + forcing_hat + control_hat
    return terms
  
  
  def control_term(self):
      """Computes the user-specified forcing term of the vorticity equation 
      Args:
        omega_hat: Fourier transformed vorticity term
        forcing: Forcing function as specified by environment or user
      """
      cfx, cfy  = self.flow.control_function
      if cfx is not None:
        kx, ky = self.grid
        cfx_hat, cfy_hat = jnp.fft.rfft2(cfx), jnp.fft.rfft2(cfy)
        # Transform the velocity forcing into vorticity 
        derivative_term = (2j * jnp.pi)
        f_vorticity = derivative_term * (cfy_hat*kx - cfx_hat*ky)
        return f_vorticity
      else:
        return None 
    
    
  def forcing_term(self):
      """Computes the user-specified forcing term of the vorticity equation 
      Args:
        omega_hat: Fourier transformed vorticity term
        forcing: Forcing function as specified by environment or user
      """
      forcing_func = self.flow.forcing_function
      if forcing_func is not None:
        kx, ky = self.grid
        x, y = self.real_grid
        fx, fy = forcing_func(k=self.flow.k, x=x, y=y)
        fx_hat, fy_hat = jnp.fft.rfft2(fx), jnp.fft.rfft2(fy)

        # Transform the velocity forcing into vorticity 
        derivative_term = (2j * jnp.pi)
        f_vorticity = derivative_term * (fy_hat*kx - fx_hat*ky)
        return f_vorticity
      else:
        return None 

class RK4CNSolver(TransientSolver):
  
  def __init__(
    self, 
    flow: FlowConfig, 
    dt: float, 
    save_n: int,
    **kwargs
  ):
    self.save_n = save_n
    self.dt = dt
    self.flow = flow 
    self.equation = PseudoSpectralNavierStokes2D(self.flow)
    super().__init__(flow, dt)
    
  def RK4_CN(self):
    
    """ Crank-Nicolson RK4 implicit-explicit time stepping scheme
        Low storage scheme inspired by [1]. Method described in [2]. 
        
        Implicit-Explicit timestepping for an ODE of the form:
          ∂u/∂t = g(u,t) + l(u,t)
        where g(u,t) is the nonlinear advection term and l(u,t) is the linear diffusion term.
        
        [1] Kochkov, D., et. al. (2021) https://doi.org/10.1073/pnas.2101784118
        [2] PK Sweby, (1984). SIAM journal on numerical analysis 21, Appendix D.
    """
    g = tree_math.unwrap(self.equation.nonlinear_terms)
    l = tree_math.unwrap(self.equation.linear_terms)
    y = tree_math.unwrap(self.equation.implicit_timestep, vector_argnums=0)

    @tree_math.wrap
    def time_step_fn(u):
      h = 0
      for k in range(5):
        h = g(u) + _beta_RK4[k] * h 
        mu = 0.5 * self.dt * (_alpha_RK4[k + 1] - _alpha_RK4[k])
        yn = u + _gammas_RK4[k]* self.dt * h + mu*l(u)
        u = y(yn, mu)
      return u
    
    return time_step_fn


  def step(self, flow: FlowConfig, dt: float, save_n: int, callbacks: Callable):
    """
    Lax.scan to iteratively apply a function given an initial value 

    Args:
        initialization(grid array): the initial fft vorticity field
        steps (int):  number of timesteps
        save_n (int): save every n steps
        ignore_intermediate_steps (bool): if saving every n steps, ignore intermediate steps.
                                          this drastically reduces the memory requirements.
        
    """
    func = self.RK4_CN()
    
    def inner_scan(initialization):
      f = lambda init, inputs: (func(init), init)
      final_state, outputs = lax.scan(f, initialization, xs=None, length=save_n)
      return final_state

    return inner_scan 
  
  def solve(self, dt: float, flow: FlowConfig, t_span: Tuple[float, float], callbacks: Iterable[CallbackBase] = [], controller: Callable = None, save_n: int = 1) -> PDEBase:

    end_time = t_span[1]
    
    if end_time < 1:
      raise ValueError("This flow configuration requires the end time to be at least 1. Please adjust your t_span value and run again.")
    
    initialization = flow.initialize_state()
    step_to_save = int(save_n // dt) 

    total_steps = end_time // dt 
    outer_steps = total_steps // step_to_save 
    
    
    inner_scan = self.step(flow, dt, step_to_save, callbacks)
    
    outer_scan = lambda init, inputs: (inner_scan(init), 
                                       inner_scan(init))
    
    final_state, outputs = lax.scan(outer_scan, initialization, xs=None, length=outer_steps)
    flow.vorticity = outputs 
    # Dummy values for iter, t for hydrogym api callback function. 
    # Optimized iteration through JAX (with scan) is not the same as native python, and the iterations can not easily be tracked.
    for cb in callbacks:
      cb(flow)
    return final_state, outputs