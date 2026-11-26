"""Collection rules for test/.

The adjoint tests (test_*_grad.py) need firedrake_adjoint, which the Firedrake venv of the images does not have; CI
skips them too. The other Firedrake-only modules skip themselves with pytest.importorskip("firedrake").
"""

import importlib.util

collect_ignore = []
if importlib.util.find_spec("firedrake_adjoint") is None:
    collect_ignore += ["test_cavity_grad.py", "test_pinball_grad.py", "test_step_grad.py"]
