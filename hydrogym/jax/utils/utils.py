import jax.numpy as jnp 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from ..flow import *
from jax import lax
from matplotlib.animation import PillowWriter

def compute_velocity_fft(omega_hat, kx, ky):
    """
    Computing the fourier velocity components (u_hat, v_hat) from the stream function (phi_hat)
    (Yin, Z. 2004)
        
    Args:
        omega_hat: the Fourier transform of the vorticity 
        grid: the jnp grid
        
    """
    double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(kx)**2 + abs(ky)**2)
    double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

    psi_hat = -1 * omega_hat / double_derivative 
    vxhat = (2 * jnp.pi * 1j) * ky * psi_hat # Get u,v from phi 
    vyhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat
    return vxhat, vyhat

def dealiasing(advection_term):
    """ 
    
    Adds the 2/3 aliasing technique to the velocity field, which 
    sets the last 1/3 high frequency Fourier modes to 0. 
    Reference: https://notes.yeshiwei.com/pseudo_spectral_method/algorithm.html

    Args:
        vel_hat: velocity field in Fourier space
    """
    n, m = advection_term.shape[0], advection_term.shape[1]
    kn, km = int(n//2 * 2//3), int(m * 2//3)
    advection_term.at[kn:2*kn, :].set(0.0)
    advection_term.at[:, km:].set(0.0)
    
    return advection_term

def compute_energy_mode(uhat, vhat, kx, ky, n, m):
    """
    Compute the energy of a specific mode and wavenumber. 
    
    Args: 
        omega_hat: fft vorticity 
        kx: wavenumber x
        ky: wavenumber y 
        n, m: grid size
    """
    
    # Compute indices of wavenumber 
    #uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    kx_idx = kx % n
    ky_idx = ky % m 
    energy = 0.5 * (jnp.abs(uhat[kx_idx, ky_idx])**2 + jnp.abs(vhat[kx_idx, ky_idx])**2) / ((n*m)**2)
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
    #uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    kx_idx = kx % n
    ky_idx = ky % m 
    velocity_mag = jnp.sqrt((jnp.abs(uhat[kx_idx, ky_idx])**2 + jnp.abs(vhat[kx_idx, ky_idx])**2) / jnp.float64(n*m)**2)
    return velocity_mag 

def compute_real_velocity_point(uhat, vhat, x_idx, y_idx):
    """
    Compute the velocity of a specific point in real space. 
    
    Args: 
        uhat, vhat: fft velocity components 
        kx: wavenumber x
        ky: wavenumber y 
        n, m: grid size
    """
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    velocity_mag = jnp.sqrt((jnp.abs(ureal[x_idx, y_idx])**2 + jnp.abs(vreal[x_idx, y_idx])**2))
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
        n: grid length
        
    """
    
    uhat, vhat = compute_velocity_fft(omega_hat, kx, ky)
    ureal = jnp.fft.irfftn(uhat)
    vreal = jnp.fft.irfftn(vhat)
    du_dy = jnp.gradient(ureal, axis=1) 
    dv_dx = jnp.gradient(vreal, axis=0)
    du_dx = jnp.gradient(ureal, axis=0)
    dv_dy = jnp.gradient(vreal, axis=1)
    avg_epsilon = 2 * nu * ((du_dx)**2 + (dv_dy)**2 + (du_dy+ dv_dx)**2)
    epsilon = jnp.sum(avg_epsilon) / (4*jnp.pi**2)
    
    return epsilon 
    
def compute_tke(omega_hat, kx, ky, n):
    """
        Computes the TKE of the systen given the fft vorticity field.
        
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
    avg_tke = 0.5 * (jnp.abs(u-U)**2 + jnp.abs(v-V)**2)  * (1/n**2)
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
    actions_cost = 75*(actions[0] + actions[1] + actions[2] + actions[3])
    return tke - actions_cost

def compute_divergence(omega_hat, kx, ky):
    """
        Computes the divergence of the systen given the fft vorticity field.
        
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
    
def create_animation(trajectory, gif_name, frame_interval_factor):
    """
        Produces an animation of the trajectory.
        
    Args: 
        trajectory: numpy file of fft vorticity trajectory
        gif_name: file name of gif file that will be saved 
        interval: frame interval as related to the length of the trajectory. 
    """
    if type(trajectory) == str:
        trajectory = jnp.load(trajectory)
    simulation = jnp.fft.irfftn(trajectory, axes=(1,2))

    fig, ax = plt.subplots()
    cax = ax.imshow(simulation[0], cmap='icefire',interpolation='nearest', vmin=-8, vmax=8)
    fig.colorbar(cax)
    
    num_frames = len(simulation)
    interval = int(num_frames * frame_interval_factor)

    timestamp = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    def update_frame(frame):
        cax.set_array(simulation[frame])
        timestamp.set_text(f'Time: {frame}')
        return cax, timestamp

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=interval)

    # Save as a GIF
    ani.save('{}.gif'.format(gif_name), writer=PillowWriter(fps=interval))