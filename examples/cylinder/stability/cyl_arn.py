"""Eigendecomposition with Arnoldi iteration"""

from cyl_common import base_checkpoint, flow, stabilization, velocity_order

flow.load_checkpoint(base_checkpoint)
qB = flow.q.copy(deepcopy=True)
