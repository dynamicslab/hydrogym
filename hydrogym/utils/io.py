import firedrake as fd

from hydrogym.flows import Flow
from typing import Any, Optional, Callable, Tuple

class CallbackBase:
    def __init__(self, interval: Optional[int] = 1):
        self.interval = interval

    def __call__(self, iter: int, t: float, state: Any):
        iostep = (iter % self.interval == 0)
        return iostep

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
            self.postprocess = lambda state: state

    def __call__(self, iter: int, t: float, state: Tuple[fd.Function]):
        if super().__call__(iter, t, state):
            state = self.postprocess(state)
            if (iter % self.interval == 0):
                self.file.write(*state, time=t)

class CheckpointCallback(CallbackBase):
    def __init__(self,
            flow: Flow,
            interval: Optional[int] = 1,
            filename: Optional[str] = 'output/checkpoint.h5'):
        super().__init__(interval=interval)
        self.filename = filename
        self.flow = flow

    def __call__(self, iter: int, t: float, state: Tuple[fd.Function]):
        if super().__call__(iter, t, state):
            self.flow.save_checkpoint(self.filename)

class GenericCallback(CallbackBase):
    def __init__(self,
            callback: Callable,
            interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback
    
    def __call__(self, iter: int, t: float, state: Tuple[fd.Function]):
        if super().__call__(iter, t, state):
            self.cb(iter, t, state)