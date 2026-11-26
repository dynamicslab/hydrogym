"""Unit tests: FlowConfig's mesh fallback.

When no ``mesh`` key is given, checkpoint auto-inference uses the mesh name
``self.DEFAULT_MESH`` (not the mesh directory path). The value is
interpolated into the auto-inferred HF checkpoint env name
``{Flow}_2D_Re{Re}_{mesh}_FD``, so it must be a name that can match an
environment on the Hub.

Called on bare instances (object.__new__) with ``_resolve_single_checkpoint``
monkeypatched to capture the inferred env name -- no firedrake flow
construction and no HF access.

pytest.importorskip guards: firedrake via the flow modules.
"""

import pytest

pytest.importorskip("firedrake")

from hydrogym.firedrake.envs.cylinder.flow import Cylinder  # noqa: E402
from hydrogym.firedrake.flow import FlowConfig  # noqa: E402


@pytest.fixture
def stub_flow_cls(monkeypatch):
    """Concrete proxy of Cylinder with checkpoint resolution captured.

    The subclass name flows into the auto-inferred env name
    (``{cls.__name__}_2D_Re{Re}_{mesh}_FD``), so assertions match on the
    suffix, not the prefix.
    """

    class _CylinderProxy(Cylinder):
        pass

    recorded = {}

    def fake_resolve_single(self, checkpoint, *args, **kwargs):
        recorded["checkpoint"] = checkpoint
        return None

    monkeypatch.setattr(FlowConfig, "_resolve_single_checkpoint", fake_resolve_single)
    return _CylinderProxy, recorded


class TestMeshFallback:
    def test_default_mesh_is_a_name_not_a_path(self, stub_flow_cls):
        """Omitting `mesh` infers an HF env name without a filesystem path."""
        cls, recorded = stub_flow_cls
        flow = object.__new__(cls)
        flow._resolve_checkpoint(restart=None, Re=100, mesh=cls.DEFAULT_MESH)
        assert "/" not in recorded["checkpoint"]
        assert recorded["checkpoint"].endswith(f"_2D_Re100_{cls.DEFAULT_MESH}_FD")

    def test_explicit_mesh_still_respected(self, stub_flow_cls):
        """Flows that pass mesh= explicitly (test_cyl.py & friends) use that mesh."""
        cls, recorded = stub_flow_cls
        flow = object.__new__(cls)
        flow._resolve_checkpoint(restart=None, Re=100, mesh="fine")
        assert recorded["checkpoint"].endswith("_2D_Re100_fine_FD")

    def test_default_mesh_attr_exists_on_all_flows(self):
        """Every firedrake flow defines DEFAULT_MESH (the fallback mesh name)."""
        from hydrogym.firedrake.envs.cavity.flow import Cavity
        from hydrogym.firedrake.envs.pinball.flow import Pinball
        from hydrogym.firedrake.envs.step.flow import Step

        for flow_cls in (Cylinder, Cavity, Pinball, Step):
            assert isinstance(flow_cls.DEFAULT_MESH, str)
            assert "/" not in flow_cls.DEFAULT_MESH
