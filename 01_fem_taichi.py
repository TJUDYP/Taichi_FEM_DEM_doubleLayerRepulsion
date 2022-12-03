import taichi as ti
import taichi.math as tm
import gmsh
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

ti.init(arch = ti.cpu)

gmsh.initialize()

gmsh.model.add("t1")

lc = 0.01

# =========================================
# make the whole region
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)

# =========================================
# make region for hole
gmsh.model.geo.addPoint(0.4, 0.4, 0, lc, 5)
gmsh.model.geo.addPoint(0.6, 0.4, 0, lc, 6)
gmsh.model.geo.addPoint(0.6, 0.6, 0, lc, 7)
gmsh.model.geo.addPoint(0.4, 0.6, 0, lc, 8)

gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)

gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)

# =========================================
#gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.addPlaneSurface([1, 2], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

entities = gmsh.model.getEntities()

# get the data of triangles and coords in mesh
for e in entities:
    if e[0] == 2:

        dim = e[0]
        tag = e[1]
        
        # Get the mesh elements for the entity (dim, tag):
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        
element_node_tags = np.reshape(elemNodeTags, (np.shape(elemTags)[1], 3))

# in gmsh, get all the nodes and coords if dim = -1 and tag = -1
dim = -1
tag = -1
nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
node_tags = nodeTags
node_coords = np.reshape(nodeCoords, (np.shape(nodeTags)[0], 3))

# get the tags of boundary nodes

nodeTags_boundary_exterior = np.array([], dtype = np.uint64)
nodeTags_boundary_interior = np.array([], dtype = np.uint64)

for e in entities:
    
    if e[0] != 2:
        dim = e[0]
        tag = e[1]        
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
        
        if e[1] <= 4:
            nodeTags_boundary_exterior = np.append(nodeTags_boundary_exterior, np.array(nodeTags, dtype=np.uint64))
        else:
            nodeTags_boundary_interior = np.append(nodeTags_boundary_interior, np.array(nodeTags, dtype=np.uint64))

node_tags_boundary_exterior = nodeTags_boundary_exterior
node_tags_boundary_interior = nodeTags_boundary_interior
node_tags_others = np.delete(node_tags, np.concatenate((node_tags_boundary_exterior-1, node_tags_boundary_interior-1)))

gmsh.clear()
gmsh.finalize()



number_points    = np.shape(node_coords)[0]
number_elements  = np.shape(element_node_tags)[0]

node_coords_taichi = ti.Vector.field(n = 3, dtype = ti.f64, shape = node_coords.shape[0])
node_coords_taichi.from_numpy(node_coords)
element_node_tags_taichi = ti.Vector.field(n = 3, dtype = ti.u64, shape = element_node_tags.shape[0])
element_node_tags_taichi.from_numpy(element_node_tags)

A_taichi = ti.field(ti.f64, shape = (number_points, number_points))
M_taichi = ti.field(ti.f64, shape = (number_points, number_points))
b_taichi = ti.field(ti.f64, shape = (number_points, 1))
loc2glb_taichi = ti.Vector.field(n = 3, dtype = ti.u64, shape = ())


@ti.func
def polygon_area(x, y) -> ti.f64:
    y_roll = ti.Vector([y[2], y[0], y[1]])
    x_roll = ti.Vector([x[2], x[0], x[1]]) 
    area = 0.5 * (tm.dot(x, y_roll)-tm.dot(y, x_roll))
    if area < 0.0:
        area *= -1
    
    return area

@ti.func
def hat_gradients(x, y) -> ti.Vector:
    area = polygon_area(x, y)
    gra_1 = ti.Vector([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / 2.0 / area
    gra_2 = ti.Vector([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / 2.0 / area
    return ti.Vector([area, gra_1, gra_2])

@ti.kernel
def stiffness_assembler_2D():
    
    for i in range(number_elements):
        loc2glb_taichi[None] = element_node_tags_taichi[i] - 1
        
        # can't using slice in taichi :-( 
        # using this method to alternative
        x1 = node_coords_taichi[loc2glb_taichi[None][0]][0]
        x2 = node_coords_taichi[loc2glb_taichi[None][1]][0]
        x3 = node_coords_taichi[loc2glb_taichi[None][2]][0]
        
        y1 = node_coords_taichi[loc2glb_taichi[None][0]][1]
        y2 = node_coords_taichi[loc2glb_taichi[None][1]][1]
        y3 = node_coords_taichi[loc2glb_taichi[None][2]][1]
        
        x = ti.Vector([x1, x2, x3])
        y = ti.Vector([y1, y2, y3])
        
        area   = hat_gradients(x, y)[0]
        grad_1 = hat_gradients(x, y)[1]
        grad_2 = hat_gradients(x, y)[2]
    
        A_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][0]] += area*(grad_1[0]*grad_1[0]+grad_2[0]*grad_2[0])
        A_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][1]] += area*(grad_1[0]*grad_1[1]+grad_2[0]*grad_2[1])
        A_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][2]] += area*(grad_1[0]*grad_1[2]+grad_2[0]*grad_2[2])
        A_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][0]] += area*(grad_1[1]*grad_1[0]+grad_2[1]*grad_2[0])
        A_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][1]] += area*(grad_1[1]*grad_1[1]+grad_2[1]*grad_2[1])
        A_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][2]] += area*(grad_1[1]*grad_1[2]+grad_2[1]*grad_2[2])
        A_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][0]] += area*(grad_1[2]*grad_1[0]+grad_2[2]*grad_2[0])
        A_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][1]] += area*(grad_1[2]*grad_1[1]+grad_2[2]*grad_2[1])
        A_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][2]] += area*(grad_1[2]*grad_1[2]+grad_2[2]*grad_2[2])

@ti.kernel
def mass_assembler_2D():
    
    for i in range(number_elements):
        loc2glb_taichi[None] = element_node_tags_taichi[i] - 1
        
        # can't using slice in taichi :-( 
        # using this method to alternative
        x1 = node_coords_taichi[loc2glb_taichi[None][0]][0]
        x2 = node_coords_taichi[loc2glb_taichi[None][1]][0]
        x3 = node_coords_taichi[loc2glb_taichi[None][2]][0]
        
        y1 = node_coords_taichi[loc2glb_taichi[None][0]][1]
        y2 = node_coords_taichi[loc2glb_taichi[None][1]][1]
        y3 = node_coords_taichi[loc2glb_taichi[None][2]][1]
        
        x = ti.Vector([x1, x2, x3])
        y = ti.Vector([y1, y2, y3])
        
        area = polygon_area(x, y)
    
        M_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][0]] += 2.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][1]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][0], loc2glb_taichi[None][2]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][0]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][1]] += 2.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][1], loc2glb_taichi[None][2]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][0]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][1]] += 1.0 / 12.0 * area
        M_taichi[loc2glb_taichi[None][2], loc2glb_taichi[None][2]] += 2.0 / 12.0 * area

@ti.func
def load_fun(x, y) -> ti.f64:    
    # specify the function on the right side of Possion function yourself. :-)
    f = tm.sin(x + y) 
    #f = 1.0   
    return f

@ti.kernel
def load_assembler_2D():
    
    for i in range(number_elements):
        loc2glb_taichi[None] = element_node_tags_taichi[i] - 1
        
        # can't using slice in taichi :-( 
        # using this method to alternative
        x1 = node_coords_taichi[loc2glb_taichi[None][0]][0]
        x2 = node_coords_taichi[loc2glb_taichi[None][1]][0]
        x3 = node_coords_taichi[loc2glb_taichi[None][2]][0]
        
        y1 = node_coords_taichi[loc2glb_taichi[None][0]][1]
        y2 = node_coords_taichi[loc2glb_taichi[None][1]][1]
        y3 = node_coords_taichi[loc2glb_taichi[None][2]][1]
        
        x = ti.Vector([x1, x2, x3])
        y = ti.Vector([y1, y2, y3])
        
        area = polygon_area(x, y)
        
        b_taichi[loc2glb_taichi[None][0], 0] += load_fun(x[0], y[0]) / 3.0 * area
        b_taichi[loc2glb_taichi[None][1], 0] += load_fun(x[1], y[1]) / 3.0 * area
        b_taichi[loc2glb_taichi[None][2], 0] += load_fun(x[2], y[2]) / 3.0 * area

      
stiffness_assembler_2D()
mass_assembler_2D()
load_assembler_2D()


A = A_taichi.to_numpy()
b = b_taichi.to_numpy()

A_00 = A[np.ix_(node_tags_others-1, node_tags_others-1)]
A_0b = np.concatenate((A[np.ix_(node_tags_others-1, node_tags_boundary_exterior-1)], \
                       A[np.ix_(node_tags_others-1, node_tags_boundary_interior-1)]), axis = 1)
b_0 = b[node_tags_others-1]
b_b = np.concatenate((np.array([[0]]*node_tags_boundary_exterior.shape[0]), \
                      np.array([[0]]*node_tags_boundary_interior.shape[0])), axis = 0)

xi = np.zeros((node_tags.shape[0], 1), dtype = np.float64)
xi[np.concatenate((node_tags_boundary_exterior-1, node_tags_boundary_interior-1))] = b_b

# xi[node_tags_others-1] = np.linalg.solve(A_00, b_0-np.matmul(A_0b, b_b))

xi[node_tags_others-1] = np.reshape(spsolve(sparse.csc_matrix(A_00), b_0-np.matmul(A_0b, b_b)), (node_tags_others.shape[0], 1))

plt.figure(figsize = (7.5, 6))
plt.scatter(node_coords[:,0], node_coords[:,1], c = xi, s=10.0, cmap='viridis')
plt.colorbar()
plt.savefig('taichi_cal.jpg', dpi=600)
plt.show()





