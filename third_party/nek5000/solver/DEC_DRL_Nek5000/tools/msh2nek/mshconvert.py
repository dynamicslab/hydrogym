#!/usr/bin python
import sys
import re
from sets import Set
from copy import copy
from numpy import zeros, array, sign, cross, dot, ones, arctan, sin, cos, pi, mod, sqrt, save
#from numpy import *
from numpy.linalg import norm
from pylab import find
from operator import add
from scipy.optimize import fsolve

#from dolfin import File, MeshEditor, Mesh, Cell, facets

print "Converting from ANSYS Fluent format (.msh) to Nek5000, semtex or FEniCS format"

# Use regular expressions to identify sections and tokens found in a fluent file
re_dimline  = re.compile(r"\(2\s(\d)\)")
re_comment  = re.compile(r"\(0\s.*")
re_zone0    = re.compile(r"\(10\s\(0\s(\w+)\s(\w+)\s(\d+)\s(\d+)\)\)")
re_zone     = re.compile(r"\(10\s\((\w+)\s(\w+)\s(\w+)\s(\d+)\s(\d)\)(\(|)")
re_face0    = re.compile(r"\(13(\s*)\(0\s+(\w+)\s+(\w+)\s+(0|0 0)\)\)")
re_face     = re.compile(r"\(13(\s*)\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)(\s*)(\(|)")
#re_periodic = re.compile(r"\(18.*\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\).*\(")
re_periodic   = re.compile(r"\(18(\s*)\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)(\s*)(\(|)")
re_pfaces   = re.compile(r"((^\s)|)(\w+)(\s*)(\w+)")
re_cells0   = re.compile(r"\(12(\s*)\(0(\s+)(\w+)(\s+)(\w+)(\s+)(0|0 0)\)\)")
re_cells    = re.compile(r"\(12.*\((\w+)\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\)\)")
re_cells2   = re.compile(r"\(12(\s*)\((\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\)(\s*)(\(|)")
re_zones    = re.compile(r"\((45|39)\s+\((\d+)\s+(\S+)\s+(\S+).*\)\((.*|[0-9]+[\.]*[0-9]*)\)\)")
re_parant   = re.compile(r"(^\s*\)(\s*)|^\s*\)\)(\s*)|^\s*\(\s*)")

# The fluent mesh (the .msh file) is basically stored as a list of nodes, and then a 
# list of faces for each zone of the mesh, the interior and the boundaries.

# Declare som maps that will be built when reading in the lists of nodes and faces:
cell_map = {}               # Maps cell id with nodes
cell_face_map = {}          # Maps cell id with faces
face_cell_map = {}          # Maps face id with two cells
face_list = []              # List of faces [[id, 2-4 nodes, 2 connecting cells and type]]
face_map = {}               # For each cell a dictionary with key=face and val=local face number
nodes = None                # Will be numpy array of nodes

# Information about connectivity and boundaries
boundary_map = {}           # For each cell and local face number, get connected cell and local face number
boundary_val_map = {}       # Holds boundary conditions, like V, v, P, etc
boundary_nodes = {}         # List of nodes attached to a boundary. Key is zone id
boundary_nodes_face_map = {}# Map of the faces that belongs to a node on a boundary
boundary_faces = {}         # List of faces attached to a boundary. Key is zone id

# Information about mesh periodicity
periodic_face_map = {}      # Map of face to face periodicity
periodic_cell_face_map = {} # Map (cell, local face number) of periodic face to (cell, local face number) of shadow
periodic_node_map = {}      # Periodic node to node map

# Some global values
num_cells = {}              # Total number of cells in different zones
zones = {}                  # zone information
zone_number_of_faces = {}   # number of faces for each zone

# For Nek5000
temperature_val_map = {}    # Boundary conditions for temperature 
passive_scalar_val_map = [] # Boundary conditions for passive scalars
face_node_map = {}          # Maps face with nodes
mid_point_map = {}          # Store location of midpoint for key face

# For Nek5000 and semtex that can have curved boundaries
curves_map = {}             # Holds curve information
curved_faces = {}           # For each boundary zone store faces that are in curved section
curved_nodes = {}           # For each boundary zone store nodes that are in curved section 

# This part is just a default that goes into the top of the rea-file:
start_of_rea = """****** PARAMETERS *****
    2.610000     NEKTON VERSION
   {0:2d} DIMENSIONAL RUN
          103 PARAMETERS FOLLOW
   1.00000         p1  DENSITY
  -100.000         p2  VISCOS
   0. 
   0. 
   0. 
   0. 
   1.00000         p7  RHOCP
   1.00000         p8  CONDUCT
   0. 
   0.              p10 FINTIME
   100           p11 NSTEPS
  -0.10000E-01     p12 DT
   0.              p13 IOCOMM
   0.              p14 IOTIME
   1000               p15 IOSTEP
   0.              p16 PSSOLVER
   0. 
  -20.00000        p18 GRID
  -1.00000         p19 INTYPE
   10.0000         p20 NORDER
   0.100000E-05    p21 DIVERGENCE
   0.100000E-06    p22 HELMHOLTZ
   0.              p23 NPSCAL
   0.100000E-01    p24 TOLREL
   0.100000E-01    p25 TOLABS
   1.00000         p26 COURANT/NTAU 
   2.00000         p27 TORDER
   0.              p28 TORDER: mesh velocity (0: p28=p27)
   0.              p29 magnetic visc if > 0, = -1/Rm if < 0
   0.              p30 > 0 ==> properties set in uservp()
   0.              p31 NPERT: #perturbation modes
   0.              p32 #BCs in re2 file, if > 0
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0.              p41 1-->multiplicative SEMG
   0.              p42 0=gmres/1=pcg
   0.              p43 0=semg/1=schwarz
   0.              p44 0=E-based/1=A-based prec.
   0.              p45 Relaxation factor for DTFS
   0.              p46 reserved
   0.              p47 vnu: mesh matieral prop
   0. 
   0. 
   0. 
   0. 
   0.              p52 IOHIS
   0. 
   0.              p54 1,2,3-->fixed flow rate dir=x,y,z
   0.              p55 vol.flow rate (p54>0) or Ubar (p54<0)
   0. 
   0. 
   0. 
   0.              p59 !=0 --> full Jac. eval. for each el.
   0.              p60 !=0 --> init. velocity to small nonzero
   0. 
   0.              p62 >0 --> force byte_swap for output
   0.              p63 =8 --> force 8-byte output
   0.              p64 =1 --> perturbation restart
   1.00000         p65 #iofiles (eg, 0 or 64); <0 --> sep. dirs
   4.00000         p66 output : <0=ascii, else binary
   4.00000         p67 restart: <0=ascii, else binary
   0.              p68 iastep: freq for avg_all
   0. 
   0. 
   0. 
   0. 
   0. 
   0.              p74 verbose Helmholtz
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   0.               p84 !=0 --> sets initial timestep if p12>0
   0.               p85 dt ratio if p84 !=0, for timesteps>0
   0.               p86 reserved
   0. 
   0. 
   0. 
   0. 
   0. 
   0. 
   20.0000          p93 Number of previous pressure solns saved
   3.00000          p94 start projecting velocity after p94 step
   5.00000          p95 start projecting pressure after p95 step
   0. 
   0. 
   0. 
   0.               p99   dealiasing: <0--> off/3--> old/4--> new
   0.              
   0.               p101   No. of additional filter modes
   1.00000          p102   Dump out divergence at each time step
  -1.00000          p103   weight of stabilizing filter (.01) 
      4  Lines of passive scalar data follows2 CONDUCT; 2RHOCP
   1.00000       1.00000       1.00000       1.00000       1.00000
   1.00000       1.00000       1.00000       1.00000
   1.00000       1.00000       1.00000       1.00000       1.00000
   1.00000       1.00000       1.00000       1.00000
           13  LOGICAL SWITCHES FOLLOW
  T     IFFLOW
  F     IFHEAT
  T     IFTRAN
  T F F F F F F F F F F IFNAV & IFADVC (convection in P.S. fields)
  F F T T T T T T T T T T IFTMSH (IF mesh for this field is T mesh)
  F     IFAXIS
  F     IFSTRS
  F     IFSPLIT
  F     IFMGRID
  F     IFMODEL
  F     IFKEPS
  F     IFMVBD
  F     IFCHAR
   5.00000       5.00000      -2.75000      -2.75000     XFAC,YFAC,XZERO,YZERO
"""

# This part goes into the bottom of the rea-file:
end_of_rea = """            0 PRESOLVE/RESTART OPTIONS  *****
            7         INITIAL CONDITIONS *****
C Default
C Default
C Default
C Default
C Default
C Default
C Default
  ***** DRIVE FORCE DATA ***** BODY FORCE, FLOW, Q
            4                 Lines of Drive force data follow
C
C
C
C
  ***** Variable Property Data ***** Overrrides Parameter data.
            1 Lines follow.
            0 PACKETS OF DATA FOLLOW
  ***** HISTORY AND INTEGRAL DATA *****
            0   POINTS.  Hcode, I,J,H,IEL
  ***** OUTPUT FIELD SPECIFICATION *****
            6 SPECIFICATIONS FOLLOW
  F      COORDINATES
  T      VELOCITY
  T      PRESSURE
  F      TEMPERATURE
  F      TEMPERATURE GRADIENT
            0      PASSIVE SCALARS
  ***** OBJECT SPECIFICATION *****
       0 Surface Objects
       0 Volume  Objects
       0 Edge    Objects
       0 Point   Objects
"""

def read_zone_nodes(dim, Nmin, Nmax, ifile):
    """Scan lines for nodes and return in an array."""
    line = ifile.readline()
    readline = False
    if re.search(re_parant, line): # check for initial paranthesis
        readline = True
        #dummy = lines.pop(0)
    global nodes 
    nodes = zeros((dim, Nmax - Nmin + 1))
    for i in range(Nmin, Nmax + 1):
        if readline:
            line = ifile.readline()
        readline = True
        nodes[:, i - Nmin] = [eval(x) for x in line.split()]

def read_periodic(ifile, periodic_dx):
    """Scan periodic section and create periodic_face_map.
    However, if a dictionary periodic_dx has been supplied then simply skip
    past this section without creating the periodic_face_map."""
    while 1:
        #line = lines.pop(0)
        line = ifile.readline()

        a = re.search(re_pfaces, line)
        if a:
            if not periodic_dx:
                periodic_face_map[int(a.group(3), 16)] = int(a.group(5), 16)
            continue
        break

    if not periodic_dx:
        keys = periodic_face_map.keys()
        vals = periodic_face_map.itervalues()
        for key, val in zip(keys, vals):
            periodic_face_map[val] = key
        
def create_periodic_face_map(periodic_dx):
    """Create a map between periodic zones.
    periodic_dx is a dictionary with key a tuple of the connected zones  
    and value (dx, dy, dz) the actual separation between the zones.
    If periodic_dx is {} the loop is never entered.
    """
    for zone0, zone1 in periodic_dx:
        print 'Generating map for periodic zones ', zone0, zone1, '\n'
        face0_list = []
        face1_list = []
        nodes0 = []
        nodes1 = []
        # Get the faces of mapped zones
        for i, face in enumerate(face_list):
            if face[-1] == zone0:
                face0_list.append((face, i))
                nodes0 += face[1]
                face[-2] = 8 # bc_type is now periodic
            elif face[-1] == zone1:
                face1_list.append((face, i))
                nodes1 += face[1]
                face[-2] = 12 # bc_type is now shadow
        # Get unique lists of nodes
        nodes0 = list(Set(nodes0))
        nodes1 = list(Set(nodes1))

        periodic_node_map[(zone0, zone1)] = {}
        # Get mapping from zone0 to zone1
        dx = array(periodic_dx[(zone0, zone1)])            

        print nodes
        print nodes0, nodes1

        print dx
        # Go through all nodes in zone0 and find the periodic match
        for node in nodes0:
            original_node = nodes[:, node - 1]

            print node, original_node
            for shadow_node in nodes1:
                if all(abs(abs(nodes[:, shadow_node - 1] - original_node) 
                           - abs(dx)) < 1.e-7*norm(dx)):
                    periodic_node_map[(zone0, zone1)][node] = shadow_node
                    nodes1.remove(shadow_node)
                    
        # Generate periodic face map
        for face, face_number in face0_list:
            nodes0 = face[1]
            true_nodes_of_shadow = [periodic_node_map[(zone0, zone1)][i] 
                                                        for i in nodes0]            
            for face_of_shadow, shadow_number in face1_list:
                nodes_of_shadow = face_of_shadow[1]
                if len(Set(nodes_of_shadow + true_nodes_of_shadow)) == 4:
                    periodic_face_map[face_number + 1] = shadow_number + 1
                    break
                    
            face1_list.remove((face_of_shadow, shadow_number))
    
def read_faces(zone_id, Nmin, Nmax, bc_type, face, ifile):
    """Read all faces and create cell_face_map + some boundary maps."""
    
    line = ifile.readline()
    readline = False
    if re.search(re_parant, line): # check for initial paranthesis
        readline = True

    ls = []
    for i in range(Nmin, Nmax + 1):
        if readline:
            line = ifile.readline()
        readline = True
        ln = line.split()
        if face == 0:
            nd = int(ln[0]) # Number of nodes
            nds = [int(x, 16) for x in ln[1:(nd + 1)]]
            cells = [int(x, 16) for x in ln[(nd + 1):]]
        else:
            nd = face
            nds = [int(x, 16) for x in ln[:nd]]
            cells = [int(x, 16) for x in ln[nd:]]
            
        face_list.append([nd, copy(nds), copy(cells), bc_type, zone_id])
        if len(nds) == 2:
            face_cell_map[(nds[0], nds[1])] = copy(cells)
            face_cell_map[(nds[1], nds[0])] = copy(cells)

        face_number = len(face_list)
        if min(cells) == 0: # A boundary zone
            if zone_id in boundary_nodes:
                boundary_nodes[zone_id] += nds
                boundary_faces[zone_id] += [face_number - 1]
                for nd in nds:
                    if nd in boundary_nodes_face_map[zone_id]:
                        boundary_nodes_face_map[zone_id][nd] += [face_number - 1]
                    else:
                        boundary_nodes_face_map[zone_id][nd] = [face_number - 1]
            else:
                boundary_nodes[zone_id] = nds
                boundary_faces[zone_id] = [face_number - 1]
                boundary_nodes_face_map[zone_id] = { nd: [face_number - 1]}

        for c in cells:
            if c > 0:                                                                      
                if not c in cell_face_map:
                    cell_face_map[c] = [face_number]
                else:
                    # Preliminary cell_face_map. Needs shaping up later
                    cell_face_map[c].append(face_number)    

    if min(cells) == 0:
        boundary_nodes[zone_id] = list(Set(boundary_nodes[zone_id]))

def create_cell_map(dim):
    """Create cell_map for use with fenics mesh."""
    for cell, faces in cell_face_map.iteritems():
        
        for face in faces:
            nds = face_list[face - 1][1]
                
            if not cell in cell_map:
                cell_map[cell] = copy(nds)
                
            else:
                cell_map[cell] = list(Set(cell_map[cell] + nds))

def create_cell_face_map(dim, mesh_format):
    """Creates maps from cells to faces and gives faces local numbering."""
    if mesh_format == 'fenics':
        create_cell_map(dim)
    else:
        eval('create_cell_face_map_' + str(dim) + 'D()') 

def create_cell_face_map_2D():
    
    for cell, faces in cell_face_map.iteritems():
        
        for face in faces:
            nds = face_list[face - 1][1]
                
            if not cell in cell_map:
                cell_map[cell] = copy(nds)
                
            elif len(cell_map[cell]) == 4:
                # Finished
                pass
            
            elif len(Set(cell_map[cell] + nds)) - len(cell_map[cell] +
                                                    nds) == 0:
                # No common node, no connectivity
                pass
            
            else:
                # Add to list preserving rotational connectivity
                # but not neccessarily right-handed
                if nds[0] in cell_map[cell]:
                    if nds[0] == cell_map[cell][0]:
                        cell_map[cell].insert(0, nds[1])
                    else:
                        cell_map[cell].append(nds[1])
                else:
                    if nds[1] == cell_map[cell][0]:
                        cell_map[cell].insert(0, nds[0])
                    else:
                        cell_map[cell].append(nds[0])

    # Ensure right-handed mesh
    for c, nds in cell_map.iteritems():
        cross_product = crss2d(nds)
        if any([i <= 0 for i in cross_product]):
            cell_map[c].reverse() 
            cross_product = crss2d(nds)
    
    # Generate face_map
    for cell, nds in cell_map.iteritems():
        face_map[cell]={(nds[0], nds[1]) : 1, (nds[1], nds[0]) : 1,
                        (nds[1], nds[2]) : 2, (nds[2], nds[1]) : 2,
                        (nds[2], nds[3]) : 3, (nds[3], nds[2]) : 3,
                        (nds[3], nds[0]) : 4, (nds[0], nds[3]) : 4,
                        1 : (nds[0], nds[1]), 1 : (nds[1], nds[0]),
                        2 : (nds[1], nds[2]), 2 : (nds[2], nds[1]),
                        3 : (nds[2], nds[3]), 3 : (nds[3], nds[2]),
                        4 : (nds[3], nds[0]), 4 : (nds[0], nds[3])}

def create_cell_face_map_3D():    
    """From GambiToNek:
    PREFERED NEKTON FORMAT:   - edge/midpt #'s shown in ( )  
         
                       7 ----(6)------- 6      face  corner points                                     
                      / .             / |      1     {1,2,6,5} 
                    (7) .           (5) |      2     {2,3,7,6}     
                    / (11)          / (10)     3     {4,3,7,8}          
                   8 ----(8)------ 5    |      4     {1,4,8,5}                     
                   |    .          |    |      5     {1,2,3,4}                     
                   |    3 .....(2).|... 2      6     {5,6,7,8}                     
                 (12)   .         (9)  /                            
                   | (3)           | (1)
                   | .             | /                                                                   
                   |.              |/                                              
                   4 ----(4)------ 1  
    """
    local_face_number = {'Bottom':1, 1:'Bottom', 
                         'East':2, 2:'East', 
                         'Top':3, 3:'Top', 
                         'West':4, 4:'West', 
                         'South':5, 5:'South', 
                         'North':6, 6:'North'}

    # Local node numbering. 
    # Note that order of points is only relevant for the Bottom face that we 
    # place first because all other faces are placed based on matching
    # nodes with the Bottom face.
    face_dict_numbering = dict(
        Bottom = array([1, 2, 6, 5], int)[::-1] - 1,
        East = array([2, 3, 7, 6], int)[::-1] - 1,
        Top = array([4, 3, 7, 8], int)[::-1] - 1,
        West = array([1, 4, 8, 5], int)[::-1] - 1,
        South = array([1, 2, 3, 4], int)[::-1] - 1,
        North = array([5, 6, 7, 8], int)[::-1] - 1
        )
            
    # Create a map from nodes of the first placed face to the remaining faces.
    # local_node_map maps nodes 1, 2, 6, 5 correctly to nodes 4,3,7,8
    # For new face 'West' node 4 is connected to node 1 of the Bottom face etc.
    # As a (somewhat irrelevant ) rule we always place the first face in Bottom.
    local_node_map = dict(West={1:4, 5:8}, 
                          East={2:3, 6:7}, 
                          Top={1:4, 2:3, 5:8, 6:7}, 
                          South={1:4, 2:3}, 
                          North={5:8, 6:7})

    # Map of opposite faces in the box.
    shadow_face = {1:3, 3:1, 2:4, 4:2, 5:6, 6:5}

    for cell, faces in cell_face_map.iteritems():
        # Place first face
        new_face_pos = 1               # Put the first face in Bottom
        face = faces[0]                # The first face from the list
        face1 = face_list[face - 1]    # Some more info for the face
        nodes1 = array(face1[1], int)  # The nodes of first face
        face_map[cell] = {}            # cell/face to local face number map
        face_map[cell][face] = new_face_pos        
        cm = cell_map[cell] = zeros(8, int)  
        newface = local_face_number[new_face_pos] # Name of new face        
        # Place the first four nodes of the cell
        cm[face_dict_numbering[newface][:]] = nodes1[:]
        nodes_of_first_face = array(cm, int)

        # One face and 4 nodes placed, now to the remaining faces
        remaining_faces = face_dict_numbering.keys()
        remaining_faces.remove(newface)   # already placed     
        faces.remove(face)                
        for face in faces: # Loop over remaining faces to be placed
            face2 = face_list[face - 1]
            nodes2 = array(face2[1], int)  # Nodes of new face
            common_nodes = find(map(lambda x: x in nodes1, nodes2))
            new_nodes = find(map(lambda x: x not in cm, nodes2))
            
            if common_nodes.shape[0] == 0:
                # e.g., opposite Bottom is Top
                opposite_face = shadow_face[new_face_pos]  
                face_map[cell][face] = opposite_face
                remaining_faces.remove(local_face_number[opposite_face])
                # Don't place any nodes because we don't know connectivity
                
            elif common_nodes.shape[0] == 2:
                matching_nodes = nodes2[common_nodes]
                pos = []
                for i, node in enumerate(matching_nodes):
                    pos.append(find(node == cm)[0])
                
                # The two common nodes determine where the face is located
                for rface in remaining_faces:
                    rval = face_dict_numbering[rface]
                    if all(map(lambda x: x in rval, pos)):
                        break # break out of loop when we find the position
                remaining_faces.remove(rface)
                
                # Give the new face its local face number in face_map
                face_map[cell][face] = local_face_number[rface] 
                
                # Now place the new nodes in cell_map
                # The connectivity of nodes is known from face_dict_numbering
                for new_node_orig_pos in new_nodes:
                    if new_node_orig_pos < 3:
                        node_to_right = nodes2[new_node_orig_pos + 1]
                    else:
                        node_to_right = nodes2[0]
                    if new_node_orig_pos > 0:
                        node_to_left = nodes2[new_node_orig_pos - 1]
                    else:
                        node_to_left = nodes2[3]
                        
                    if any(node_to_right == nodes_of_first_face):
                        # if the node to the right is already placed
                        nd = find(node_to_right == nodes_of_first_face)[0]
                        cm[local_node_map[rface][nd + 1] - 1] =  \
                                                      nodes2[new_node_orig_pos]
                        
                    else:
                        nd = find(node_to_left == nodes_of_first_face)[0]
                        cm[local_node_map[rface][nd + 1] - 1] = \
                                                      nodes2[new_node_orig_pos]
                            
        if any(x < 0. for x in crss(cm)):
            print "HHHHHHHHHHHHHHH",cell, cm
            # We guessed the wrong direction for first face. Reverse direction.
            cm_copy = copy(cm)
            cm[[0, 1, 5, 4]] = cm_copy[[4, 5, 1, 0]]
            cm[[3, 2, 6, 7]] = cm_copy[[7, 6, 2, 3]]
            
            # We need change the face map
            fm = face_map[cell]
            
            for k, v in fm.iteritems( ) :
                if v == 5 :
                    fm[k] = 6
                elif v == 6 :
                    fm[k] = 5

#            print face_map[cell]
        
#            if cell == 7 :
#                face_map[cell] = {32: 2, 2: 1, 5: 4, 7: 5, 15: 3, 25: 6}
#            else :
#                face_map[cell] = {1: 1, 3: 5, 7: 4, 13: 3, 20: 2, 30: 6}


def crss2d(nds):
    """Test that we have the correct orientation for 2D mesh."""
    cross_product = []
    nds = nodes[:, array(nds) - 1]
    for i, j, k in zip([2, 3, 1, 4], [4, 1, 3, 2], [1, 2, 4, 3]):
        xy1 = nds[:, i - 1]
        xy2 = nds[:, j - 1]
        xy0 = nds[:, k - 1]
        v1x = xy1[0] - xy0[0]
        v2x = xy2[0] - xy0[0]
        v1y = xy1[1] - xy0[1]
        v2y = xy2[1] - xy0[1]
        cross_product.append(v1x*v2y - v1y*v2x)
    return cross_product
    
def crss(nds):
    """Test that we have the correct orientation for 3D mesh."""
    cross_product = []
    nds = nodes[:, array(nds) - 1]
    for i, j, k, l in zip([2, 3, 1, 4, 6, 7, 5, 8], [4, 1, 3, 2, 8, 5, 7, 6], 
                          [5, 6, 8, 7, 1, 2, 4, 3], [1, 2, 4, 3, 5, 6, 8, 7]):
        p1 = nds[:, i - 1]
        p2 = nds[:, j - 1]
        p3 = nds[:, k - 1]
        p0 = nds[:, l - 1]        
        u = p1 - p0
        v = p2 - p0
        w = p3 - p0        
        c = array([u[1]*v[2] - u[2]*v[1],
                   u[2]*v[0] - u[0]*v[2],
                   u[0]*v[1] - u[1]*v[0]])
        cross_product.append(dot(w, c))  
        
    for i in range(4, 8):
        cross_product[i] = - cross_product[i]
        
    return cross_product
            
def create_periodic_cell_face_map():
    """Create maps of type local face 1 of cell 2 is periodic to local face 2 
    of cell 10."""
    for f0, f1 in periodic_face_map.iteritems():
        # f0, f1 = periodic face0 - face1
        face0 = face_list[f0 - 1]
        face1 = face_list[f1 - 1] # shadow
        nd, nds, cells, bc_type, zone_id = [0,]*2, [0,]*2, [0,]*2, [0,]*2, [0,]*2
        for i, ff in enumerate([face0, face1]):
            nd[i], nds[i], cells[i], bc_type[i], zone_id[i] = ff
        
        cell_face_pair = []
        for i in range(2):
            c = max(cells[i])
            if len(nds[i]) == 2:
                cell_face_pair.append((c, face_map[c][(nds[i][0], nds[i][1])]))
            else:
                cell_face_pair.append((c, face_map[c][eval('f'+str(i))]))
                
        periodic_cell_face_map[cell_face_pair[0]] = cell_face_pair[1]
        periodic_cell_face_map[cell_face_pair[1]] = cell_face_pair[0]
                   
def create_boundary_section(bcs, temperature, passive_scalars, mesh_format):
    
    if mesh_format == 'fenics':
        return
        
    for i in range(len(passive_scalars)):
        passive_scalar_val_map.append({})
                        
    global bcs_copy
    if not bcs:
        bcs_copy = {}
    else:
        bcs_copy = bcs
    
    for face_number, face in enumerate(face_list):
        nd, nds, cells, bc_type, zone_id = face

        if min(cells) == 0: # Boundary face
            c = max(cells)
            if len(nds) == 2:
                local_face  = face_map[c][(nds[0], nds[1])]
            else:
                local_face  = face_map[c][face_number + 1]
            if bc_type == 8 or bc_type == 12:
                # Face is periodic.
                boundary_map[(c, local_face)] = \
                                    periodic_cell_face_map[(c, local_face)]                                   
                boundary_val_map[(c, local_face)] = 'P'
                if temperature:
                    temperature_val_map[(c, local_face)] = 'P'
                for ss, scalar in enumerate(passive_scalars):
                    passive_scalar_val_map[ss][(c, local_face)] = 'P' 
                        
            else:
                boundary_map[(c, local_face)] = (0, 0)
                if bcs:
                    boundary_val_map[(c, local_face)] = bcs[zone_id]
                else:
                    boundary_val_map[(c, local_face)] = zones[zone_id][1][-1]
                    bcs_copy[zone_id] = zones[zone_id][1][-1]
                if temperature:
                    temperature_val_map[(c, local_face)] =  \
                                                    temperature[zone_id]
                for ss, scalar in enumerate(passive_scalars):
                    passive_scalar_val_map[ss][(c, local_face)] = \
                                            passive_scalars[ss][zone_id] 

        else:
            c0 = cells[0]
            c1 = cells[1]
            if len(nds) == 2:
                local_face0  = face_map[c0][(nds[0], nds[1])]
                local_face1  = face_map[c1][(nds[0], nds[1])]
            else: #3D
                local_face0 = face_map[c0][face_number + 1]
                local_face1 = face_map[c1][face_number + 1] 
                
            if bc_type == 2:
                # interior
                boundary_map[(c0, local_face0)] = (c1, local_face1)
                boundary_map[(c1, local_face1)] = (c0, local_face0)
                boundary_val_map[(c0, local_face0)] = 'E'
                boundary_val_map[(c1, local_face1)] = 'E'
                
                if temperature:
                    temperature_val_map[(c0, local_face0)] =  'E'
                    temperature_val_map[(c1, local_face1)] =  'E'
                
                for ss, scalar in enumerate(passive_scalars):
                    passive_scalar_val_map[ss][(c0, local_face0)] = 'E'
                    passive_scalar_val_map[ss][(c1, local_face1)] = 'E'
            
            else:
                raise NotImplementedError
            
def circle_center(x, x0, x1, r):
    return (r**2 - (x[0] - x0[0])**2 - (x[1] - x0[1])**2, 
            r**2 - (x[0] - x1[0])**2 - (x[1] - x1[1])**2)

def read_curved_sides(curves):
    
    for curve_zone, curve in curves.iteritems():
        cell_face_m = []
        curve['x'] = []
        spline = []
        for face_number in boundary_faces[curve_zone]:
            face = face_list[face_number]
            cell = max(face[2])
            nds = face[1]
            cell_face_m.append((cell, face_map[cell][(nds[0], nds[1])], face_number)) 
            if curve['type'] == 'C':
                for jj in range(curve['depth']):
                    # Put circle on opposite face
                    if jj == 0:
                        if 'circle_center' in curve:
                            cc_center = curve['circle_center']
                        else:
                            cc_center = fsolve(circle_center, (1e-3, 1e-3), 
                                                args=(nodes[:, nds[0] - 1], 
                                                    nodes[:, nds[1] - 1], 
                                                curve['radius']), xtol=1e-12)
                        curve['x'].append([curve['radius'] , 0, 0, 0, 0])
                        new_radius = curve['radius']
                    opposite_face_number = mod(face_map[cell][(nds[0], nds[1])] + 1, 4) + 1
                    cell_face_m.append((cell, opposite_face_number, 0))
                    nn = opposite_nodes = face_map[cell][opposite_face_number]
                    new_radius = sign(new_radius)*0.5*(sqrt((nodes[0, nn[0] - 1] - cc_center[0])**2 + 
                                        (nodes[1, nn[0] - 1] - cc_center[1])**2) +
                                    sqrt((nodes[0, nn[1] - 1] - cc_center[0])**2 + 
                                        (nodes[1, nn[1] - 1] - cc_center[1])**2))
                    curve['x'].append([-new_radius, 0, 0, 0, 0])
                    # Put circle on opposing cell
                    opp_cell = list(face_cell_map[(nn[0], nn[1])])
                    opp_cell.remove(cell)
                    opp_cell = opp_cell[0]
                    opposite_face_number = face_map[opp_cell][(nn[0], nn[1])]
                    cell_face_m.append((opp_cell, opposite_face_number, 0))
                    curve['x'].append([new_radius, 0, 0, 0, 0])
                    cell = opp_cell
                    nds = copy(nn)
            elif curve['type'] == 'spline':
                spline += face[1]
                    
        # Check for linearity using both faces of a boundary node.
        # In the case of not linear, add the nodes/faces to the 
        # curved_nodes/faces lists.    
        curved_faces[curve_zone] = []
        curved_nodes[curve_zone] = []

        for nd, facel in boundary_nodes_face_map[curve_zone].iteritems():
            if len(facel) > 1:
                nds_left = face_list[facel[0]][1]
                nds_right = face_list[facel[1]][1]
                n1 = nodes[:, nds_left[0]-1]
                n2 = nodes[:, nds_left[1]-1]                
                n3 = nodes[:, nds_right[1]-1]
                #if (abs(n3[0]-n2[0]) < 1e-12 and abs(n3[1]-n2[1]) < 1e-12 or 
                #    abs(n3[0]-n1[0]) < 1e-12 and abs(n3[1]-n1[1]) < 1e-12 ) :
                #    n3 = nodes[:, nds_right[0]-1]

                dnx1 = n2[0] - n1[0]
                dny1 = n2[1] - n1[1]
                dnx2 = n3[0] - n2[0]
                dny2 = n3[1] - n2[1]

                #print nd, facel, nds_left, nds_right,abs(dny1),abs(dny2),abs(dnx1),abs(dnx2),abs(dnx2/dnx1-dny2/dny1) 

                if ((abs(dny1) < 1e-6 and abs(dny2) < 1e-6) or
                    (abs(dnx1) < 1e-6 and abs(dnx2) < 1e-6)):
                    pass
                elif ((abs(dny1) < 1e-6 and not abs(dny2) < 1e-6) or 
                    (abs(dnx1) < 1e-6 and not abs(dnx2) < 1e-6) or
                    (abs(dnx2/dnx1 - dny2/dny1) > 1.e-6)): 
                    # Curved line. Add nodes to curved_nodes
                    curved_faces[curve_zone] += facel
                    curved_nodes[curve_zone] += nds_left
                    curved_nodes[curve_zone] += nds_right
        curved_faces[curve_zone] = list(Set(curved_faces[curve_zone])) 
        curved_nodes[curve_zone] = list(Set(curved_nodes[curve_zone]))
        if curve['type'] == 'C':
            pass
        elif curve['type'] == 'm':
            find_mid_point(curve_zone, curve)
        elif curve['type'] == 'spline':
            curve['x'] = list(Set(spline))            
        curves_map[curve_zone] = (cell_face_m, curve)
        
def barycentric_weight(y):
    N = len(y)
    w = ones(N, float)
    for j in range(1, N):
        for k in range(0, j):
            w[k] = (y[k] - y[j])*w[k]
            w[j] *= (y[j] - y[k])
    w = 1./w
    return w

def fun(x0, x1, y0, y1, xx, yy):
    """w are barycentric weights.
    x is unknown.
    x0, y0 contain x and y in the four nodes used to compute the midpoint."""    

    # Look for point of intersection between interpolated curve between nodes in x, y
    # and the normal to the face between nodes (x0, y0) and (x1, y1)
    # Transform coordinate axes
    # Center of face is xs, ys
    xs = (x0 + x1)/2.
    ys = (y0 + y1)/2.

    if abs(y1 - y0) > abs(x1 - x0):
        theta = arctan((x1 - x0)/(y1 - y0))
        theta2 = arctan((xx - xs)/(yy - ys))
        dy = (yy - ys)/cos(theta2)
        xn = copy(xx)
        yn = copy(yy)
        xn = dy*sin(theta2 - theta)
        yn = dy*cos(theta2 - theta)
        w = barycentric_weight(yn)
        y2 = - yn
        f = zeros(len(y2), float)
        ss = sum(w/y2)
        f[:] = w/y2/ss
        dy = dot(f, xn)
        xny = xs + dy*sin(theta + pi/2.)
        yny = ys + dy*cos(theta + pi/2.)

    else:        
        theta = arctan((y1 - y0)/(x1 - x0))
        theta2 = arctan((yy - ys)/(xx - xs))
        dx = (xx - xs)/cos(theta)
        xn = copy(xx)
        yn = copy(yy)
        xn = dx*cos(theta2 - theta)
        yn = dx*sin(theta2 - theta)
        w = barycentric_weight(xn)
        x2 = - xn
        f = zeros(len(x2), float)
        ss = sum(w/x2)
        f[:] = w/x2/ss
        dy = dot(f, yn)
        xny = xs + dy*cos(theta + pi/2.)
        yny = ys + dy*sin(theta + pi/2.)
    
    return xny, yny

def find_mid_point(curve_zone, curve):
    
    # Get unique list of nodes on boundary
    nodes0 = boundary_nodes[curve_zone]
    face0_list = boundary_faces[curve_zone]

    # Loop over faces and compute midpoint using at least two 
    # nodes of two neighbouring faces
    for i in face0_list:
        face = face_list[i] 
        nds = face[1]
        face_node_map[i] = {'left': None, 'right': None}
        for j in face0_list:
            face2 = face_list[j] 
            if len(Set(nds + face2[1])) == 3:
                if nds[0] in face2[1]:
                    face_node_map[i]['left'] = j
                else:
                    face_node_map[i]['right'] = j                   

    for face in face_node_map:
        if face in curved_faces[curve_zone]:
            fl, fr = face_node_map[face]['left'], face_node_map[face]['right']
            if fl is not None and fr is not None:
                common_node_left = map(lambda x: x in face_list[face][1], face_list[fl][1])            
                node0 = face_list[fl][1][common_node_left.index(False)]
                node1 = face_list[fl][1][common_node_left.index(True)]
                common_node_right = map(lambda x: x in face_list[face][1], face_list[fr][1])            
                node2 = face_list[fr][1][common_node_right.index(True)]
                node3 = face_list[fr][1][common_node_right.index(False)]        
                x, y = nodes[:, array([node0, node1, node2, node3]) - 1]            
                x0, y0 = x[1], y[1]
                x1, y1 = x[2], y[2]            
                
            elif fl is None:
                common_node_right = map(lambda x: x in face_list[face][1], face_list[fr][1])            
                node0 = face_list[face][1][common_node_right.index(True)]
                node1 = face_list[fr][1][common_node_right.index(True)]
                node2 = face_list[fr][1][common_node_right.index(False)]        
                x, y = nodes[:, array([node0, node1, node2]) - 1]        
                x0, y0 = nodes[:, array(face_list[face][1]) - 1]            
                x0, x1 = x0[:]
                y0, y1 = y0[:]
                            
            elif fr is None:
                common_node_left = map(lambda x: x in face_list[face][1], face_list[fl][1])            
                node0 = face_list[face][1][common_node_left.index(True)]
                node1 = face_list[fl][1][common_node_left.index(True)]
                node2 = face_list[fl][1][common_node_left.index(False)]        
                x, y = nodes[:, array([node0, node1, node2]) - 1]
                x0, y0 = nodes[:, array(face_list[face][1]) - 1]
                x0, x1 = x0[:]
                y0, y1 = y0[:]
                    
            xn, yn = fun(x0, x1, y0, y1, x, y)                
            mid_point_map[face] = (xn, yn, 0, 0, 0)


#def scan_fluent_mesh(lines):
def scan_fluent_mesh(ifile):  
    """Scan fluent mesh and generate numerous maps."""
    # Warning! Not yet tested for multiple interior zones
    dim = 0
    one = 0
    num_faces = 0
    while 1:
        line = ifile.readline()
        if len(line) == 0:
            print 'Finished reading file\n'
            break

        #try:
            #line = lines.pop(0)
        #except:
            #print 'Finished reading file\n'
            #break
        if dim == 0: # Dimension usually comes first
            a = re.search(re_dimline, line)
            if a: 
                print 'Reading dimensions\n'
                dim = int(a.group(1))
                print 'Mesh is ' + str(dim) + 'D\n'
                continue
        
        if one == 0: # The total number of nodes
            a = re.search(re_zone0, line)
            if a:
                print 'Reading zone info\n'
                one, num_vertices, dummy1, dummy2 = int(a.group(1)), \
                     int(a.group(2), 16), int(a.group(3), 16), int(a.group(4))
                continue
            
        a = re.search(re_zone, line) # Nodes
        if a:
            zone_id, first_id, last_id, dummy1, dummy2 = int(a.group(1), 16), \
                int(a.group(2), 16), int(a.group(3), 16), int(a.group(4)), \
                int(a.group(5))
            print 'Reading ', last_id - first_id + 1,' nodes in zone ', zone_id + 1, '\n'
            #read_zone_nodes(dim, first_id, last_id, lines)
            #lines = lines[(last_id - first_id + 1):]  
            read_zone_nodes(dim, first_id, last_id, ifile)
            continue
            
        a = re.search(re_zones,line) # Zone info
        if a:
            print 'Reading zone ', line
            dummy, zone_id, zone_type, zone_name, radius =  \
                       int(a.group(1)), int(a.group(2)),  a.group(3), \
                       a.group(4), a.group(5)
            zones[zone_id] = [zone_type, zone_name, radius]
            continue
        
        a = re.search(re_cells0, line) # Get total number of cells/elements
        if a:
            print 'Reading cell info ', line
            first_id, tot_num_cells = int(a.group(3),16), int(a.group(5), 16)
            continue

        a = re.search(re_cells,line) # Get the cell info.
        if a:

            zone_id, first_id, last_id, bc_type, element_type = \
                int(a.group(1),16), int(a.group(2), 16), int(a.group(3), 16), \
                int(a.group(4), 16), int(a.group(5), 16)
            print 'Reading ', last_id - first_id + 1,' cells in zone ', zone_id, '\n'
            if last_id == 0:
                raise TypeError("Zero elements!")
            num_cells[zone_id] = [first_id, last_id, bc_type, element_type]
            continue

        a = re.search(re_cells2,line) # Get the cell info.
        if a:
            raise TypeError("Wrong cell type. Can only handle one single cell type")

        a = re.search(re_face0, line)
        if a:
            print 'Reading total number of faces\n', line
            num_faces = int(a.group(3),16)
            continue
            
        a = re.search(re_face, line)
        if a:
            print 'Reading faces ', line
            zone_id, first_id, last_id, bc_type, face_type = \
                 int(a.group(2), 16), int(a.group(3), 16), int(a.group(4), 16), \
                 int(a.group(5), 16), int(a.group(6), 16)
            read_faces(zone_id, first_id, last_id, bc_type, face_type, ifile)

            #lines = lines[(last_id - first_id + 1):]
            zone_number_of_faces[zone_id] = last_id - first_id + 1
            continue
        
        a = re.search(re_periodic, line)
        if a:
            print 'Reading periodic connectivity\n', line
            read_periodic(ifile, periodic_dx)
            continue
        
        print 'Line = ',line
        if any([re.search(st, line) for st in (re_parant, re_comment)]) or \
                                                             not line.strip():
            continue
                
        # Should not make it here
        print 'Line = ',line
        raise IOError('Something went wrong reading fluent mesh.')
    
def write_nek5000_file(dim, ofilename, curves, temperature, passive_scalars):
    tot_num_cells = len(cell_map)
    ofile  = open(ofilename + '.rea', "w")
    ## Put the mesh in a rea-file
    print 'Create the rea-file: %s\n' %(ofilename+'.rea')
    print 'Writing the elements info\n'
    ofile.write(start_of_rea.format(dim))
    ofile.write(' **MESH DATA**\n')
    ofile.write('       %s       %s       %s      NEL,NDIM,NELV\n' 
                                    %(tot_num_cells, dim, tot_num_cells))
    element_header = '      ELEMENT          %s [    1 ]    GROUP     0\n'    
    for i in range(tot_num_cells):
        ofile.write(element_header %(i + 1))
        if dim == 2:
            n1 = nodes[0, array(cell_map[i + 1]) - 1]
            n2 = nodes[1, array(cell_map[i + 1]) - 1]
            ofile.write(reduce(add, ['{0:.8e}'.format(x).rjust(16) 
                                    for x in n1]) + '\n')
            ofile.write(reduce(add, ['{0:.8e}'.format(x).rjust(16) 
                                    for x in n2]) + '\n')                                                       
        else:
            nn = nodes[:, array(cell_map[i + 1]) - 1]
            for i in range(3):
                ofile.write(reduce(add, ['{0:.8e}'.format(x).rjust(16) 
                                        for x in nn[i, :4]]) + '\n')
            for i in range(3):
                ofile.write(reduce(add, ['{0:.8e}'.format(x).rjust(16) 
                                        for x in nn[i, 4:]]) + '\n')
    print 'Writing the curved side info\n'
    ofile.write("  ***** CURVED SIDE DATA *****  \n")
    cc = "{0:6d} Curved sides follow IEDGE,IEL,CURVE(I),I=1,5, CCURVE \n"
    if tot_num_cells < 1000:
        c1 = "{0:3d}{1:3d}"
    else:
        c1 = "{0:2d}{1:6d}"
    c2 = "{0:14.6e}{1:14.6e}{2:14.6e}{3:14.6e}{4:14.6e} {5:s}\n"    
    N = 0
    for zone in curves:
        if curves_map[zone][1]['type'] == 'C':
            N += len(curves_map[zone][0])
        elif curves_map[zone][1]['type'] == 'm':
            N += len(curved_faces[zone])
    ofile.write(cc.format(N))
    for zone in curves:
        if curves_map[zone][1]['type'] == 'C':
            for i in range(len(curves_map[zone][0])):
                xx = curves_map[zone][1]['x'][i]
                ofile.write(c1.format(curves_map[zone][0][i][1], 
                                    curves_map[zone][0][i][0]))
                ofile.write(c2.format(xx[0], xx[1], xx[2], xx[3], xx[4],
                                    curves_map[zone][1]['type']))
        elif curves_map[zone][1]['type'] == 'm':
            for i in range(zone_number_of_faces[zone]):     
                if curves_map[zone][0][i][2] in curved_faces[zone]:
                    ofile.write(c1.format(curves_map[zone][0][i][1], 
                                        curves_map[zone][0][i][0]))             
                    xx = mid_point_map[curves_map[zone][0][i][2]]
                    ofile.write(c2.format(xx[0], xx[1], xx[2], xx[3], xx[4],
                                        curves_map[zone][1]['type']))

    print 'Writing the boundary and bottom section\n'
    ofile.write('  ***** BOUNDARY CONDITIONS ***** \n')
    ofile.write('  ***** FLUID   BOUNDARY CONDITIONS ***** \n')
    f_str = " {0:s}  "
    if tot_num_cells < 1000:
        f_str1 = "{0:3d}{1:3d}"
    elif tot_num_cells < 100000:
        f_str1 = "{0:5d}{1:1d}"
    elif tot_num_cells < 1000000:
        f_str1 = "{0:6d}{1:1d}"    
    f_str2 = "{0:14.7e}{1:14.7e}{2:14.7e}{3:14.7e}{4:14.7e}\n"

    for i in range(1, tot_num_cells + 1):
        for j in range(1, 2*dim + 1):
            bm = boundary_map[(i, j)]
            ofile.write(f_str.format(boundary_val_map[(i, j)]))
            ofile.write(f_str1.format(i, j))
            ofile.write(f_str2.format(bm[0], bm[1], 0, 0, 0))
        
    if not temperature:
        ofile.write('  ***** NO THERMAL BOUNDARY CONDITIONS ***** \n')
    else:
        ofile.write('  ***** THERMAL BOUNDARY CONDITIONS ***** \n')
        for i in range(1, tot_num_cells + 1):
            for j in range(1, 2*dim + 1):
                bm = boundary_map[(i, j)]
                ofile.write(f_str %(temperature_val_map[(i, j)], i, j, bm[0],
                                    bm[1], 0, 0, 0))    
    if len(passive_scalars) == 0:
        pass
    else:
        for si, scalar in enumerate(passive_scalars):
            ofile.write(' ***** PASSIVE SCALAR %s  BOUNDARY CONDITIONS ***** \n' 
                        %(si + 1))
            for i in range(1, tot_num_cells + 1):
                for j in range(1, 2*dim + 1):
                    bm = boundary_map[(i, j)]
                    ofile.write(f_str %(passive_scalar_val_map[si][(i, j)], i, 
                                        j, bm[0], bm[1], 0, 0, 0))
    
    ofile.write(end_of_rea)
    print 'Finished writing the rea-file\n'
    # Close files
    ofile.close()

def write_semtex_file(dim, ofilename, curves, cylindrical, NZ):
    tot_num_cells = len(cell_map)
    if not dim == 2:
        print 'Error, semtex uses only 2D meshes'
        raise TypeError

    print 'Writing semtex mesh file\n'
    print 'Writing first some default tokens. Be sure to modify these to your own liking later\n'
    ofile  = open(ofilename, "w")
    ofile.write('<FIELDS>\n')
    if NZ == 1:
        ofile.write('    u    v    p\n')
    else:
        ofile.write('    u    v    w    p\n')
    ofile.write('</FIELDS>\n\n')
    #Inserting some default tokens:
    ofile.write("""<TOKENS>
KINVIS      = 0.01
D_T         = 0.01
N_STEP      = 100
CYLINDRICAL = %d
CHKPOINT    = 1
N_TIME      = 2
Lz          = 10
N_P         = 9
N_Z         = %d
IO_CFL      = 20
IO_FLD      = N_STEP
</TOKENS>\n\n""" %(cylindrical, NZ))
    if NZ == 1:
        ofile.write("""<USER>
    u = 0.0
    v = 0.0
    p = 0.0
</USER>\n\n""")
    else:
        ofile.write("""<USER>
    u = 0.0
    v = 0.0
    w = 0.0
    p = 0.0
</USER>\n\n""")
    semtex_bnds = {}
    for key, val in boundary_val_map.iteritems():
        if not val == 'E':
            if val in semtex_bnds:
                semtex_bnds[val].append(key)
            else:
                semtex_bnds[val] = [key]
    N = 0
    for key, val in semtex_bnds.iteritems():
        if key == 'P':
            N += len(val)/2
        else:
            N += len(val)        
    len_bnds = len(semtex_bnds) - {True: 1, False: 0}['P' in semtex_bnds]
    if len_bnds > 0:
        ofile.write('<GROUPS NUMBER=%s>\n' %(len_bnds))
        bcsc = copy(bcs_copy)
        for key, val in bcs_copy.iteritems():
            bcsc[val] = key
        groups_str = '{0:4d}{1:>8}{2:>16}\n'
        i = 0
        for key, val in semtex_bnds.iteritems():
            if not key == 'P':
                i += 1
                ofile.write(groups_str.format(i, key, zones[bcsc[key]][0]))
        ofile.write('</GROUPS>\n\n')
        ofile.write('<BCS NUMBER=%s>' %(len_bnds))
        if NZ == 1:
            dirichlet_str = """    
{0:4d}{1:>4}    3
    <D> u = 0.0     </D>
    <D> v = 0.0     </D>
    <H> p = 0.0     </H>\n"""
            neuman_str = """    
{0:4d}{1:>4}    3
    <N> u = 0.0     </N>
    <N> v = 0.0     </N>
    <D> p = 0.0     </D>\n"""
            axis_str = """    
{0:4d}{1:>4}    3
    <A> u = 0.0     </A>
    <A> v = 0.0     </A>
    <A> p = 0.0     </A>\n"""
        else:
            dirichlet_str = """    
{0:4d}{1:>4}    4
    <D> u = 0.0     </D>
    <D> v = 0.0     </D>
    <D> w = 0.0     </D>
    <H> p = 0.0     </H>\n"""
            neuman_str = """    
{0:4d}{1:>4}    4
    <N> u = 0.0     </N>
    <N> v = 0.0     </N>
    <N> w = 0.0     </N>
    <D> p = 0.0     </D>\n"""
            axis_str = """    
{0:4d}{1:>4}    4
    <A> u = 0.0     </A>
    <A> v = 0.0     </A>
    <A> w = 0.0     </A>
    <A> p = 0.0     </A>\n"""
        i = 0
        for key, val in semtex_bnds.iteritems():
            if not key == 'P':
                i += 1
                if key in ('a', 'A'):
                    ofile.write(axis_str.format(i, key))
                elif key in ('o', 'O'):
                    ofile.write(neuman_str.format(i, key))
                else:
                    ofile.write(dirichlet_str.format(i, key))
        ofile.write('</BCS>\n\n')
                                
    print 'Writing nodes\n'
    nodes_header = '<NODES NUMBER=%s>\n'
    node_str = ' {0:5d} {1:14.7e} {2:14.7e} 0.000000\n'
    ofile.write(nodes_header %(nodes.shape[1]))
    for i in range(nodes.shape[1]):
        ofile.write(node_str.format(i+1, nodes[0, i], nodes[1, i]))
        
    ofile.write('</NODES>\n\n')
    print 'Writing elements\n'
    element_header = '<ELEMENTS NUMBER=%s>\n'
    element_str = ' {0:5d} <Q> {1:4d}{2:4d}{3:4d}{4:4d} </Q>\n'
    ofile.write(element_header %(tot_num_cells))
    for i in range(1, tot_num_cells + 1):
        ofile.write(element_str.format(i, cell_map[i][0], cell_map[i][1], cell_map[i][2], cell_map[i][3]))
    ofile.write('</ELEMENTS>\n\n')
    print 'Writing surfaces\n'
    surfaces_header = '<SURFACES NUMBER=%s>\n'
    ofile.write(surfaces_header %(N))
    surfaces_str = '{0:4d}{1:4d}{2:4d} <B> {3:s} </B>\n'
    surfaces_pstr = '{0:4d}{1:4d}{2:4d} <P> {3:4d}{4:4d} </P>\n'
    i = 0
    for key in semtex_bnds.iterkeys():
        for val in semtex_bnds[key]:
            i += 1
            if not key == 'P':
                ofile.write(surfaces_str.format(i, val[0], val[1], key))
            else:
                pp = periodic_cell_face_map[val]
                ofile.write(surfaces_pstr.format(i, val[0], val[1], pp[0], pp[1]))
                semtex_bnds['P'].remove(pp)
    ofile.write('</SURFACES>\n\n') 
    if len(curves) > 0:
        print 'Writing curves\n'
        cc = '<CURVES NUMBER={}>\n'
        c1 = '{0:4d}{1:6d}{2:6d} <ARC> {3:14.7f} </ARC>\n'
        c2 = ' {0:4d} {1:6d} {2:4d}  <SPLINE> {3:s} </SPLINE>\n'
        tmp_list = []
        count = 0
        for zone in curves:
            if curves_map[zone][1]['type'] == 'C':
                for i in range(len(curves_map[zone][0])):
                    xx = curves_map[zone][1]['x'][i]
                    count += 1
                    tmp_list.append(c1.format(count, curves_map[zone][0][i][0], 
                                    curves_map[zone][0][i][1], xx[0]))
                    
            elif curves_map[zone][1]['type'] == 'spline':
                spline_file = open(ofilename + '.geom', 'w')                                            
                spl_nodes = []
                for nd in boundary_nodes[zone]:
                    if nd in curved_nodes[zone]:
                        spl_nodes.append([nodes[0, nd - 1], nodes[1, nd - 1]])
                spl_nodes = array(spl_nodes)
                ind = spl_nodes[:, 0].argsort()
                for k in ind:
                    x, y = spl_nodes[k, :]
                    spline_file.write(' {0:14.6f} {1:14.6f}\n'.format(x, y))
                spline_file.close()
                for i in range(len(curves_map[zone][0])):
                    face = face_list[curves_map[zone][0][i][2]]
                    nds = face[1]
                    if curved_nodes[zone].__contains__(nds[0] or nds[1]):
                        count += 1
                        tmp_list.append(c2.format(count, curves_map[zone][0][i][0], curves_map[zone][0][i][1], ofilename + '.geom'))
        ofile.write(cc.format(count))
        for tmp in tmp_list: ofile.write(tmp)                        
        ofile.write('</CURVES>\n')
        
    print 'Finished writing semtex mesh\n'
    ofile.close()

def write_fenics_file(dim, ofilename):
    ofile  = File(ofilename + '.xml')
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, dim, dim)
    editor.init_vertices(nodes.shape[1])
    editor.init_cells(len(cell_map))
    
    for i in range(nodes.shape[1]):
        if dim == 2:
            editor.add_vertex(i, nodes[0, i], nodes[1, i])
        else:
            editor.add_vertex(i, nodes[0, i], nodes[1, i], nodes[2, i])
            
    for i in range(1, len(cell_map)+1):
        if dim == 2:
            editor.add_cell(i-1, cell_map[i][0]-1, cell_map[i][1]-1, cell_map[i][2]-1)
        else:
            editor.add_cell(i-1, cell_map[i][0]-1, cell_map[i][1]-1, cell_map[i][2]-1, cell_map[i][3]-1)
    
    mesh.order()
    mvc = mesh.domains().markers(dim-1)
    for zone, faces in boundary_faces.iteritems():
        for face in faces:
            cell = face_list[face][2][0]
            dolfin_cell = Cell(mesh, cell-1)
            nodes_of_cell = dolfin_cell.entities(0)
            nodes_of_face = array(face_list[face][1]) - 1
            for jj, ff in enumerate(facets(dolfin_cell)):
                facet_nodes = ff.entities(0)
                if all(map(lambda x: x in nodes_of_face, facet_nodes)):
                    local_index = jj
                    break
            mvc.set_value(cell-1, local_index, zone)
        
    ofile << mesh        
    from dolfin import plot
    plot(mesh, interactive=True)
    print 'Finished writing FEniCS mesh\n'

def convert(fluentmesh, 
            func=None, 
            mesh_format='nek5000',                     # nek5000, semtex or fenics
            periodic_dx={}, curves = {}, bcs = False,  # nek5000 and semtex
            temperature=False, passive_scalars=[],     # nek5000 only
            cylindrical=1, NZ=1):                      # semtex  only
    """Converts a fluent mesh to a mesh format that can be used by Nek5000,
       semtex or FEniCS. 
         
         fluentmesh = fluent mesh (*.msh file)
         
               func = Optional function of spatial coordinates (x,y) that can 
                      be used to modify the fluent mesh.
                      For example, say you have a mesh that is a rectangular 
                      geometry with -2 < x < 6 and 0 < y 0.5. Now you want to
                      squeeze this mesh around x = 0 to create a stenosis type
                      of mesh. This can be accomplished by squeezing the mesh
                      in the y-direction through:
                      
                      def func_y(x, y):
                          if abs(x) < 1.:
                              return y*(1 - 0.25*(1 + cos(x*pi)))
                          else:
                              return y
                              
                      and supplying this value to the keyword func like:
                      
                      func={'y': func_y}
                      
                      Otherwise you might just create this stenosis in your
                      mesh generation software and supply nothing for func.
                      Note that in nek5000 you will most likely do this in
                      subroutines userdat or userdat2.
                      
        mesh_format = 'nek5000', 'semtex' or 'fenics'

                bcs = False or dictionary of boundary conditions for 
                      velocity/pressure (something like {1: 'W', 2: 'v'} for
                      wall in zone 1 and Dirchlet to be specified in zone 2).
                      False indicates that dictionary is not used and
                      in that case we assume the name of the zone ends in
                      the correct letter, like 'somename_W' for zone 1 and 
                      'somename_v' for zone 2. Zonenames are easy to modify
                      at the bottom of the fluent msh files.
                      Don't include periodic zones here.
                
        periodic_dx = Dictionary describing any periodicity in the mesh.
                      Keys are tuples of the two periodic zones (zone0, zone1)
                      and values are displacement vectors.
                      Note that the current program also can read a mesh where
                      periodicity has been generated in the mesh generator. 
                      However, this author has still not managed to 
                      create a periodic 3D mesh correctly in any mesh software.
                      Hence I prefer to define periodicity through this
                      dictionary here and do nothing regarding periodicity in
                      the meshing software. All you need to know are the ids
                      and displacement of the periodic zones. Connectivity
                      will then be computed here. Note that each periodic zone
                      needs to be its own zone. Simply creating a 3D UnitCube
                      in gambit and not assigning names to the 6 zones won't
                      work. We need zone identifiers (for now).
                      Not for FEniCS.
                      
             curves = Dictionary of curve information. Keys are curve zones 
                      and value is either 
                               {'type': 'C', 
                                'radius': radius, 
                                'circle_center': (x, y), 
                                'depth': depth} 
                      for a circle or 
                               {'type': 'm'} 
                      for midpoint with nek5000 or
                               {'type': 'spline'}
                      for a curved side with semtex. Here a .geom file containing
                      the spline information will be created.
                      
                      The circle may provide the radius or the center of the
                      circle through 'circle_center'. The curvature may also be 
                      used in the internal elements inside the surface through
                      specifying the depth. depth=4 means that the curvature 
                      is used throughout the first four elements inside that 
                      surface. This is necessary to get good quality meshes in,
                      e.g., a cylinder. The radius for an internal face is 
                      allways computed as the distance to the circle_center.
                      Not for FEniCS.
                      
        temperature = False or dictionary of boundary conditions for 
                      temperature (something like {1: 'W', 2: 'v'}) for
                      Wall in zone 1 and Dirchlet specified in .usr in zone 2.
                      False indicates that temperature is not used. 
                      (nek5000 only)
                      
    passive_scalars = [] or list of dictionaries of boundary conditions for
                      passive scalars. Empty means no passive scalars are used.
                      (nek5000 only)
                                                                  
        cylindrical = 1 for cylindrical mesh and 0 for regular (semtex only)
        
                 NZ = Number of planes in homogeneous direction (semtex only)
                 
    """
    ofilename = fluentmesh[:-4]
    ifile  = open(fluentmesh, "r")

    if not nodes:
        # Read all lines of fluent mesh
        #lines = ifile.readlines()
        #if len(lines) == 0:
            #raise IOError("Empty fluent mesh file")
        
        #scan_fluent_mesh(lines)
        scan_fluent_mesh(ifile)

    dim = nodes.shape[0]
    create_cell_face_map(dim, mesh_format)
    create_periodic_face_map(periodic_dx)
    create_periodic_cell_face_map()
    create_boundary_section(bcs, temperature, passive_scalars, mesh_format)
    # Modify the entire mesh using the shape-function func
    if func:
        sz = nodes.shape
        for i in range(sz[1]):
            x, y = nodes[:, i]
            if 'x' in func:
                xnew = func['x'](x, y)
                if abs(xnew - x) > 1e-6:
                    nodes[0, i] = xnew
            if 'y' in func:
                ynew = func['y'](x, y)
                if abs(ynew - y) > 1e-6:
                    nodes[1, i] = ynew
        if mesh_format == 'nek5000':
            print 'Warning!! Consider using userdat/userdat2 instead!'

    if not mesh_format == 'fenics':
        read_curved_sides(curves)

    # Generate the mesh files for given mesh format
    if mesh_format == 'nek5000':
        write_nek5000_file(dim, ofilename, curves, temperature, passive_scalars)
    elif mesh_format == 'semtex':
        write_semtex_file(dim, ofilename, curves, cylindrical, NZ)
    if mesh_format == 'fenics':
        write_fenics_file(dim, ofilename)

    ifile.close()
