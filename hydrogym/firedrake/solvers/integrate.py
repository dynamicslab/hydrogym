from .bdf_ext import LinearizedBDF, SemiImplicitBDF

__all__ = ["integrate"]

METHODS = {
    "BDF": SemiImplicitBDF,
    "LinearizedBDF": LinearizedBDF,
}


def integrate(
    flow,
    t_span,
    dt,
    method="BDF",
    callbacks=[],
    controller=None,
    collect_rewards=False,
    **options,
):
    """Integrate the flow forward in time with the chosen transient method.

    All transient methods share `core.TransientSolver.solve`, which supports
    optional reward collection: pass `collect_rewards=True` to get back a
    `(flow, rewards)` tuple instead of just the final flow state.
    """
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {METHODS.keys()}")

    solver = METHODS[method](flow, dt, **options)
    return solver.solve(t_span, callbacks=callbacks, controller=controller, collect_rewards=collect_rewards)
