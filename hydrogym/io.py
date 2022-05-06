import numpy as np
import firedrake as fd

from .core import FlowConfig, CallbackBase
from .utils import print
from typing import Any, Optional, Callable, Tuple

class ParaviewCallback(CallbackBase):
    def __init__(self,
            interval: Optional[int] = 1,
            filename: Optional[str] = 'output/solution.pvd',
            postprocess: Optional[Callable] = None):
        super().__init__(interval=interval)
        self.file = fd.File(filename)

        # Postprocess will be called before saving (use to compute vorticity, for instance)
        self.postprocess = postprocess
        if self.postprocess is None:
            self.postprocess = lambda flow: (flow.u, flow.p)

    def __call__(self, iter: int, t: float, flow: FlowConfig):
        if super().__call__(iter, t, flow):
            state = self.postprocess(flow)
            if (iter % self.interval == 0):
                self.file.write(*state, time=t)

class CheckpointCallback(CallbackBase):
    def __init__(self,
            interval: Optional[int] = 1,
            filename: Optional[str] = 'output/checkpoint.h5'):
        super().__init__(interval=interval)
        self.filename = filename

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if super().__call__(iter, t, flow):
            flow.save_checkpoint(self.filename)

class LogCallback(CallbackBase):
    def __init__(self,
            postprocess: Callable,
            nvals,
            interval: Optional[int] = 1,
            filename: Optional[str] = None,
            print_fmt: Optional[str] = None
            ):
        super().__init__(interval=interval)
        self.filename = filename
        self.print_fmt = print_fmt
        self.data = np.zeros((1, nvals+1))

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if super().__call__(iter, t, flow):
            new_data = np.array([t, *self.postprocess(flow)], ndmin=2)
            if iter==0:
                self.data[0, :] = new_data
            else:
                self.data = np.append(self.data, new_data, axis=0)

            if self.filename is not None:
                np.savetxt(self.filename, self.data)
            if self.print_fmt is not None:
                print(self.print_fmt.format(*new_data))

class SnapshotCallback(CallbackBase):
    def __init__(self,
            interval: Optional[int] = 1,
            output_dir: Optional[str] = 'snapshots'
        ):
        super().__init__(interval=interval)
        self.output_dir = output_dir

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if iter == 0:
            flow.save_checkpoint(f'{self.output_dir}/checkpoint.h5')
        if super().__call__(iter, t, flow):
            # TODO: Pickle and store PETSc array
            pass

class GenericCallback(CallbackBase):
    def __init__(self,
            callback: Callable,
            interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback
    
    def __call__(self, iter: int, t: float, flow: FlowConfig):
        if super().__call__(iter, t, flow):
            self.cb(iter, t, flow)