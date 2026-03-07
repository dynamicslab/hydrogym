from typing import Callable, Optional

import numpy as np

from hydrogym.core import CallbackBase

from .utils import is_rank_zero as is_rank_zero_func, print as print_func


class LogCallback(CallbackBase):

  def __init__(
      self,
      postprocess: Callable,
      nvals: int,
      interval: Optional[int] = 1,
      filename: Optional[str] = None,
      print_fmt: Optional[str] = None,
  ):
    super().__init__(interval=interval)
    self.postprocess = postprocess
    self.filename = filename
    self.print_fmt = print_fmt
    self.data = np.zeros((1, nvals + 1))

  def __call__(self, iter: int, t: float, env):
    if super().__call__(iter, t, env):
      new_data = np.array([t, *self.postprocess(env)], ndmin=2)
      if iter == 0:
        self.data[0, :] = new_data
      else:
        self.data = np.append(self.data, new_data, axis=0)

      if self.filename is not None and is_rank_zero_func():
        np.savetxt(self.filename, self.data)
      if self.print_fmt is not None:
        print_func(self.print_fmt.format(*new_data.ravel()))


class CheckpointCallback(CallbackBase):

  def __init__(
      self,
      interval: Optional[int] = 1,
      filename: Optional[str] = "checkpoint",
  ):
    super().__init__(interval=interval)
    self.filename = filename

  def __call__(self, iter: int, t: float, env):
    if super().__call__(iter, t, env):
      # For nek, checkpointing is typically handled by the environment itself
      # This is a placeholder that can be extended if needed
      if hasattr(env, 'save_checkpoint'):
        env.save_checkpoint(self.filename)


class GenericCallback(CallbackBase):

  def __init__(self, callback: Callable, interval: Optional[int] = 1):
    super().__init__(interval=interval)
    self.cb = callback

  def __call__(self, iter: int, t: float, env):
    if super().__call__(iter, t, env):
      self.cb(iter, t, env)
