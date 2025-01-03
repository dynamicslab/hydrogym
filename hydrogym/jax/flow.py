import jax.numpy as jnp
from jax import grad, lax
from hydrogym.core import PDEBase
from .utils import utils

class FlowConfig(PDEBase):
    
    DEFAULT_REYNOLDS = 200
    DEFAULT_WAVENUMBER = 4
    DEFAULT_GRID_SIZE = (64, 64)
    DEFAULT_DOMAIN_X = (0, 2*jnp.pi)
    DEFAULT_DOMAIN_Y = (0, 2*jnp.pi)
    DEFAULT_OBS_SIZE = 8 # This correlates to a total observation size of 8x8 = 64.

    def __init__(self, **config):
        self.k = config.get("k", self.DEFAULT_WAVENUMBER)
        self.Re = config.get("Re", self.DEFAULT_REYNOLDS)
        self.grid_size = config.get("grid_size", self.DEFAULT_GRID_SIZE)
        self.domain_x = config.get("domain_x", self.DEFAULT_DOMAIN_X)
        self.domain_y = config.get("domain_y", self.DEFAULT_DOMAIN_Y)
        self.obs_size = config.get("obs_size", self.DEFAULT_OBS_SIZE)
        self.control_function = ( jnp.zeros_like(self.load_mesh()[0]), jnp.zeros_like(self.load_mesh()[1]) )
        
    def load_mesh(self): 
        """
            Create jax grid given the desired dimensions and spacing in real space
            
            Returns:
                jax meshgrid 
        """
        x0, xn, nx = self.domain_x[0], self.domain_x[1], self.grid_size[1]
        y0, yn, ny = self.domain_y[0], self.domain_y[1], self.grid_size[0]
        x = jnp.linspace(x0, xn, nx)
        y = jnp.linspace(y0, yn, ny)
        return jnp.meshgrid(x, y, indexing='ij')
    
    
    def _calculate_velocity_point(self, state, k1, k2):
        # Calculate velocity point 
        kx, ky = self.load_fft_mesh()
        uhat, vhat = utils.compute_velocity_fft(state, kx, ky)
        point_velocity = utils.compute_real_velocity_point(uhat, vhat, k1, k2)
        
        return point_velocity 
    
    def state(self) -> jnp.array:
        return self.state
    
    def get_observations(self, state) -> jnp.array:
        
        n, m = self.grid_size
        divisor = n // self.obs_size 
        
        def calculate_velocity(state):
            
            points = [
                self._calculate_velocity_point(state, x, y)
                for x in range(0, n, int(n/divisor))
                for y in range(0, m, int(m/divisor))
            ]
            return jnp.array(points)

        def scan_fn(carry, state):
            obs_val = calculate_velocity(state) # To use energy observation, swap this with calculate_energy
            return carry, obs_val
            
        _, all_obs = lax.scan(scan_fn, None, state)
                        
        return jnp.mean(all_obs, axis=0)
    
    def load_fft_mesh(self):
        """Create jax grid given desired dimensions and spacing in real Fourier space

        Returns:
            jax meshgrid 
        """
        N = self.grid_size[0]
        M = self.grid_size[1]
        dx = self.domain_x[1] / N
        dy = self.domain_y[1] / M
        kx = jnp.fft.fftfreq(N, dx)
        ky = jnp.fft.rfftfreq(M, dy) 
        return jnp.meshgrid(kx, ky, indexing='ij')
    
    def initialize_state(self):
        """Generate a divergence free velocity field to initialize the state
        Initializing with divergence free field specified with the following stream function:
        
        φ(x,y) = sin(x)cos(y)

        Returns:
            fft vorticity field 
        """
        X, Y = self.load_mesh()
        # Gradients of φ(x,y) #
        def dstream_func_dx(x, y):
            return jnp.cos(x)

        def dstream_func_dy(x, y):
            return - jnp.sin(y)

        dudy = grad(dstream_func_dy, argnums=1)
        dvdx = grad(dstream_func_dx, argnums=0)
        du_dy = jnp.vectorize(dudy)(X, Y)
        dv_dx = jnp.vectorize(dvdx)(X, Y)
        vorticity  = dv_dx - du_dy
        vorticity_0 = jnp.fft.rfftn(vorticity)
        
        self._vorticity = vorticity_0
        return self._vorticity
    
    def set_BCs(self):
        # Set the boundary conditions 
        pass
    
    def forcing_function(self, k, x, y):
        """Sinusoidal forcing function that drives the Kolmogorov flow.

        Args:
            k (int): forcing wavenumber 
            x (jnp.array): spatial coordinates in x
            y (jnp.array): spatial coordinates in y

        Returns:
            tuple: forcing function in (x,y)
        """
        return ( jnp.sin(k*y), jnp.zeros_like(y) )
    
    @property
    def nu(self):
        return 1 / self.Re 
    
    @property
    def num_inputs(self) -> int:
        """Length of the control vector (number of actuators)"""
        pass

    @property
    def num_outputs(self) -> int:
        """Number of scalar observed variables"""
        pass
    
    def save_checkpoint(self):
        """Set up mesh, function spaces, state vector, etc"""
        pass

    def init_bcs(self):
        """Initialize any boundary conditions for the PDE."""
        pass
    
    def copy_state(self, deepcopy=True):
        """Return a copy of the flow state"""
        pass
    
    def evaluate_objective(self):
        """Return a copy of the flow state"""
        pass
    
    def render(self, **kwargs):
        """Plot the current PDE state (called by `gym.Env`)"""
        pass
    
    def load_checkpoint(self, filename: str):
        pass

