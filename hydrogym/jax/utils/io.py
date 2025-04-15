import jax.numpy as jnp 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from jax import lax
from matplotlib.animation import PillowWriter

from hydrogym.core import CallbackBase
from typing import Callable, Optional, Tuple 

class LogCallback(CallbackBase):
    
    def __init__(
        self,
        postprocess: Callable,
        interval: Optional[int] = 1,
        filename: Optional[str] = None,
        print_fmt: Optional[str] = None 
    ):
        super().__init__(interval=interval)
        self.filename = filename
        self.postprocess = postprocess
        self.print_fmt = print_fmt
        
    def __call__(self, flow):
        self.postprocess(flow)
        np.savetxt(self.filename, self.postprocess(flow), fmt="%.5f") 
                
      
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
    cax = ax.imshow(simulation[0], cmap='viridis',interpolation='nearest', vmin=-8, vmax=8)
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