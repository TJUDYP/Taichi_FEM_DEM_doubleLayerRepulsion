import taichi as ti
import taichi.math as tm
import gmsh
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch = ti.cpu)


# ===================================================================
# using gmsh to generate mesh in following Finete element analysis
# ===================================================================
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
gmsh.model.geo.addPlaneSurface([1, 2], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# =========================================
# get the boundary node tags to give Dirichlet or Neumann boundary condition
for i in range(8):
    gmsh.model.mesh.create_edges(dimTags=[(1, i+1)])

edgeTags, edgeNodes = gmsh.model.mesh.getAllEdges()
edgeTags  = edgeTags.reshape(-1, 1)
edgeNodes = edgeNodes.reshape(-1, 2)

edge = np.concatenate((edgeTags, edgeNodes), axis=1)
node_tags_boundary = edge[np.argsort(edge[:,0])]

# =========================================
# get the data of triangles'tags in mesh
entities = gmsh.model.getEntities()
for e in entities:
    if e[0] == 2:

        dim = e[0]
        tag = e[1]
        
        # Get the mesh elements for the entity (dim, tag):
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
        
element_node_tags = np.reshape(elemNodeTags, (np.shape(elemTags)[1], 3))

# =========================================
# in gmsh, get all the nodes and coords if dim = -1 and tag = -1
dim = -1
tag = -1
nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
node_tags = nodeTags
node_coords = np.reshape(nodeCoords, (np.shape(nodeTags)[0], 3))

# =========================================
gmsh.clear()
gmsh.finalize()


# ===================================================================
# using taichi to finish the Finete element analysis
# ===================================================================

number_points    = np.shape(node_coords)[0]
number_elements  = np.shape(element_node_tags)[0]

node_coords_taichi = ti.Vector.field(n = 3, dtype = ti.f64, shape = node_coords.shape[0])
element_node_tags_taichi = ti.Vector.field(n = 3, dtype = ti.i32, shape = element_node_tags.shape[0])
node_tags_boundary_taichi = ti.Vector.field(n = 3, dtype = ti.i32, shape = node_tags_boundary.shape[0])

node_coords_taichi.from_numpy(node_coords)
element_node_tags_taichi.from_numpy(element_node_tags.astype(np.int32))
node_tags_boundary_taichi.from_numpy(node_tags_boundary.astype(np.int32))

A = ti.linalg.SparseMatrixBuilder(number_points, number_points, max_num_triplets= 20 * number_points)
R = ti.linalg.SparseMatrixBuilder(number_points, number_points, max_num_triplets= 20 * number_points)
b_taichi = ti.field(ti.f64, shape = (number_points, 1))
r_taichi = ti.field(ti.f64, shape = (number_points, 1))

loc2glb_taichi = ti.Vector.field(n = 3, dtype = ti.i32, shape = ())

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
def stiffness_assembler_2D(A: ti.types.sparse_matrix_builder()):
    
    for i in range(number_elements):
        
        loc2glb_taichi[None] = element_node_tags_taichi[i] - 1

        n1 = loc2glb_taichi[None][0]
        n2 = loc2glb_taichi[None][1]
        n3 = loc2glb_taichi[None][2]
        
        # can't using slice in taichi :-( 
        # using this method to alternative
        x1 = node_coords_taichi[n1][0]
        x2 = node_coords_taichi[n2][0]
        x3 = node_coords_taichi[n3][0]
        
        y1 = node_coords_taichi[n1][1]
        y2 = node_coords_taichi[n2][1]
        y3 = node_coords_taichi[n3][1]
        
        x = ti.Vector([x1, x2, x3])
        y = ti.Vector([y1, y2, y3])
        
        area   = hat_gradients(x, y)[0]
        grad_1 = hat_gradients(x, y)[1]
        grad_2 = hat_gradients(x, y)[2]
    
        A[n1, n1] += area*(grad_1[0]*grad_1[0]+grad_2[0]*grad_2[0])
        A[n1, n2] += area*(grad_1[0]*grad_1[1]+grad_2[0]*grad_2[1])
        A[n1, n3] += area*(grad_1[0]*grad_1[2]+grad_2[0]*grad_2[2])
        A[n2, n1] += area*(grad_1[1]*grad_1[0]+grad_2[1]*grad_2[0])
        A[n2, n2] += area*(grad_1[1]*grad_1[1]+grad_2[1]*grad_2[1])
        A[n2, n3] += area*(grad_1[1]*grad_1[2]+grad_2[1]*grad_2[2])
        A[n3, n1] += area*(grad_1[2]*grad_1[0]+grad_2[2]*grad_2[0])
        A[n3, n2] += area*(grad_1[2]*grad_1[1]+grad_2[2]*grad_2[1])
        A[n3, n3] += area*(grad_1[2]*grad_1[2]+grad_2[2]*grad_2[2])

@ti.kernel
def robin_mass_matrix_2D(R: ti.types.sparse_matrix_builder(), kappa: ti.f64):

    for i in range(node_tags_boundary.shape[0]):
        
        n1 = int(node_tags_boundary_taichi[i][1] - 1)
        n2 = int(node_tags_boundary_taichi[i][2] - 1)
        
        x1 = node_coords_taichi[n1][0]
        x2 = node_coords_taichi[n2][0]
        
        y1 = node_coords_taichi[n1][1]
        y2 = node_coords_taichi[n2][1]
        
        length = tm.sqrt((x1-x2)**2+(y1-y2)**2)
        
        k = kappa
        
        R[n1, n1] += k / 6 * 2 * length
        R[n1, n2] += k / 6 * 1 * length
        R[n2, n1] += k / 6 * 1 * length
        R[n2, n2] += k / 6 * 2 * length   

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
                   
        n1 = ti.cast(loc2glb_taichi[None][0], ti.i32)
        n2 = ti.cast(loc2glb_taichi[None][1], ti.i32)
        n3 = ti.cast(loc2glb_taichi[None][2], ti.i32)
        
        x1 = node_coords_taichi[n1][0]
        x2 = node_coords_taichi[n2][0]
        x3 = node_coords_taichi[n3][0]
        
        y1 = node_coords_taichi[n1][1]
        y2 = node_coords_taichi[n2][1]
        y3 = node_coords_taichi[n3][1]
        
        x = ti.Vector([x1, x2, x3])
        y = ti.Vector([y1, y2, y3])
        
        area = polygon_area(x, y)
        
        b_taichi[n1, 0] += load_fun(x[0], y[0]) / 3.0 * area
        b_taichi[n2, 0] += load_fun(x[1], y[1]) / 3.0 * area
        b_taichi[n3, 0] += load_fun(x[2], y[2]) / 3.0 * area

@ti.kernel
def robin_load_vector_2D(kappa: ti.f64, gD: ti.f64, gN: ti.f64):
    
    for i in range(node_tags_boundary.shape[0]):
        
        n1 = int(node_tags_boundary_taichi[i][1] - 1)
        n2 = int(node_tags_boundary_taichi[i][2] - 1)
        
        x1 = node_coords_taichi[n1][0]
        x2 = node_coords_taichi[n2][0]
        
        y1 = node_coords_taichi[n1][1]
        y2 = node_coords_taichi[n2][1]
        
        length = tm.sqrt((x1-x2)**2+(y1-y2)**2)
        
        tmp = kappa * gD + gN
        
        r_taichi[n1, 0] += tmp * 1 * length / 2
        r_taichi[n2, 0] += tmp * 1 * length / 2

# =========================================
# using sparse matrix in taichi to compute the result
stiffness_assembler_2D(A)
A_taichi = A.build()

robin_mass_matrix_2D(R, 1000000.0)
R_taichi = R.build()

load_assembler_2D()
robin_load_vector_2D(1000000.0, 0.0, 0.0)

A_solver = A_taichi + R_taichi
b_solver = b_taichi.to_numpy() + r_taichi.to_numpy()

solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A_solver)
solver.factorize(A_solver)
xi = solver.solve(b_solver)

# =========================================
plt.figure(figsize = (7.5, 6))
plt.scatter(node_coords[:,0], node_coords[:,1], c = xi, s = 10.0, cmap='viridis')
plt.colorbar()

plt.savefig('taichi_cal.jpg', dpi=600)
plt.show()

print("节点总数：", np.shape(node_coords)[0])    
print("三角形单元总数：", np.shape(element_node_tags)[0])
