import firedrake as fd

from .core import FlowConfig, CallbackBase
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

class LogfileCallback(CallbackBase):
    def __init__(self,
            interval: Optional[int] = 1,
            filename: Optional[str] = 'output/log.dat',
            postprocess: Optional[Callable] = None
            ):
        super().__init__(interval=interval)
        self.filename = filename

class GenericCallback(CallbackBase):
    def __init__(self,
            callback: Callable,
            interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback
    
    def __call__(self, iter: int, t: float, flow: FlowConfig):
        if super().__call__(iter, t, flow):
            self.cb(iter, t, flow)