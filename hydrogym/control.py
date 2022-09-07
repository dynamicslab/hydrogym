from typing import Callable, Optional

from .core import CallbackBase, PDEModel


# TODO
class ControllerCallback(CallbackBase):
    def __init__(self, callback: Callable, interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback

    def __call__(self, iter: int, t: float, flow: PDEModel):
        if super().__call__(iter, t, flow):
            self.cb(iter, t, flow)
