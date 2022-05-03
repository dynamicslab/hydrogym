from . import utils
from . import flows
from .timestepping import IPCSSolver
from .utils import io

import os
install_dir = os.path.abspath(f'{__file__}/..')