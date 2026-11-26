"""The mesh requested by name is the mesh that is loaded (regression: FlowConfig popped 'mesh' before PDEBase)."""

import pytest

fd = pytest.importorskip("firedrake")
import hydrogym.firedrake as hgym  # noqa: E402


@pytest.mark.parametrize("mesh", ["medium", "fine"])
def test_pinball_loads_requested_mesh(mesh):
    flow = hgym.Pinball(Re=100, mesh=mesh, use_HF_data_manager=False)
    ref = fd.Mesh(f"{flow.MESH_DIR}/{mesh}.msh", name="mesh")
    assert flow.mesh.num_cells() == ref.num_cells()
