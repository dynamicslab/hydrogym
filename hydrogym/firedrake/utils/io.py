from typing import Callable, Optional, Tuple

import firedrake as fd
import numpy as np
from firedrake import logging

from hydrogym.core import CallbackBase, PDEBase

from .utils import is_rank_zero, print


class ParaviewCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
        filename: Optional[str] = "output/solution.pvd",
        postprocess: Optional[Callable] = None,
    ):
        super().__init__(interval=interval)
        self.file = fd.File(filename)

        # Postprocess will be called before saving (use to compute vorticity, for instance)
        self.postprocess = postprocess
        if self.postprocess is None:
            self.postprocess = lambda flow: (flow.u, flow.p)

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            state = self.postprocess(flow)
            if iter % self.interval == 0:
                self.file.write(*state, time=t)


class CheckpointCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
        filename: Optional[str] = "output/checkpoint.h5",
        write_mesh=True,
    ):
        super().__init__(interval=interval)
        self.filename = filename
        self.write_mesh = write_mesh

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if super().__call__(iter, t, flow):
            flow.save_checkpoint(self.filename, write_mesh=self.write_mesh)


class LogCallback(CallbackBase):
    def __init__(
        self,
        postprocess: Callable,
        nvals,
        interval: Optional[int] = 1,
        filename: Optional[str] = None,
        print_fmt: Optional[str] = None,
    ):
        super().__init__(interval=interval)
        self.postprocess = postprocess
        self.filename = filename
        self.print_fmt = print_fmt
        self.data = np.zeros((1, nvals + 1))

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if super().__call__(iter, t, flow):
            new_data = np.array([t, *self.postprocess(flow)], ndmin=2)
            if iter == 0:
                self.data[0, :] = new_data
            else:
                self.data = np.append(self.data, new_data, axis=0)

            if self.filename is not None and is_rank_zero():
                np.savetxt(self.filename, self.data)
            if self.print_fmt is not None:
                print(self.print_fmt.format(*new_data.ravel()))


class SnapshotCallback(CallbackBase):
    def __init__(
        self, interval: Optional[int] = 1, filename: Optional[str] = "snapshots"
    ):
        """
        Save snapshots as checkpoints for modal analysis

        Note that this slows down the simulation
        """
        super().__init__(interval=interval)
        self.h5 = fd.CheckpointFile(filename, "w")
        self.snap_idx = 0
        self.saved_mesh = False

    def __call__(self, iter: int, t: float, flow: Tuple[fd.Function]):
        if super().__call__(iter, t, flow):
            if not self.saved_mesh:
                self.h5.save_mesh(flow.mesh)
                self.saved_mesh = True
            self.h5.save_function(flow.q, idx=self.snap_idx)
            self.snap_idx += 1

    def close(self):
        logging.log(logging.DEBUG, "Closing snapshot CheckpointFile")
        self.h5.close()


class GenericCallback(CallbackBase):
    def __init__(self, callback: Callable, interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback

    def __call__(self, iter: int, t: float, flow: PDEBase):
        if super().__call__(iter, t, flow):
            self.cb(iter, t, flow)
