from .bdf_ext import SemiImplicitBDF
from .ipcs import IPCS

__all__ = ["integrate"]

METHODS = {
    "IPCS": IPCS,
    "BDF": SemiImplicitBDF,
}


def integrate(flow, t_span, dt, method="BDF", callbacks=[], controller=None, **options):
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")

    solver = METHODS[method](flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks, controller=controller)
