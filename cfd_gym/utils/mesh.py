import meshio

def load_mesh(mesh_dir):
    import fenics
    
    """Load mesh (precomputed with mesh-convert.py)"""
    mesh = fenics.Mesh()
    with fenics.XDMFFile(f"{mesh_dir}/mesh.xdmf") as infile:
        infile.read(mesh)

    mvc = fenics.MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1) 
    with fenics.XDMFFile(f"{mesh_dir}/mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    mf = fenics.cpp.mesh.MeshFunctionSizet(mesh, mvc)
    return mesh, mf

def msh_to_xdmf(mesh, cell_type, prune_z=False):
    """Convert gmsh file to FEniCS-readable XDMF"""
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

def convert_to_xdmf(msh_file, out_dir='.', dim=2):
    assert dim in [2, 3]
    if dim == 2:
        element, boundary = "triangle", "line"
        prune_z = True
    elif dim == 3:
        element, boundary = "tetra", "triangle"
        prune_z = False
        
    msh = meshio.read(msh_file)
    meshio.write(f"{out_dir}/mesh.xdmf", msh_to_xdmf(msh, element, prune_z))
    meshio.write(f"{out_dir}/mf.xdmf", msh_to_xdmf(msh, boundary, prune_z))