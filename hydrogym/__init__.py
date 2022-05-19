from . import (
    utils,
    flow,
    ts,
    env,
    control
)
from .ts import integrate
from .utils import io, linalg, print, is_rank_zero

import os
install_dir = os.path.abspath(f'{__file__}/..')

# Convenience imports from firedrake and UFL
from firedrake import (
    logging,
    Function,
    CheckpointFile,
    project
)
from ufl import inner, dot, grad, nabla_grad, div, dx, ds, curl