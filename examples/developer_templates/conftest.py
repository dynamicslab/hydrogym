"""Make each skeleton directory import its own modules when pytest is run
from the repo root (`pytest examples/developer_templates/`)."""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _add_skeleton_dirs_to_path():
    here = os.path.dirname(__file__)
    for sub in ("minimal_inprocess_solver", "minimal_external_solver"):
        path = os.path.join(here, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    yield
