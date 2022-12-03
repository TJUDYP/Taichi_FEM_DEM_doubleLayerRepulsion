import gmsh
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


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
# gmsh.model.geo.addPlaneSurface([1], 1)
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


element_tags = np.reshape(elemTags, np.shape(elemTags)[1])
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





def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y,1))-np.dot(y, np.roll(x,1)))

def hat_gradients(x, y):
    area  = polygon_area(x, y)
    gra_1 = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / 2.0 / area
    gra_2 = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / 2.0 / area
    return [area, gra_1, gra_2]

def stiffness_assembler_2D(node_coords, element_node_tags):
    number_points    = np.shape(node_coords)[0]
    number_elements  = np.shape(element_node_tags)[0]
    
    # sparse matrix can be used here to optimize ?
    # sparse matrix is much slower in scipy when get or storage data, so I choose to use numpy here.
    A = np.zeros((number_points, number_points))
    
    for i in range(number_elements):
        loc2glb = element_node_tags[i, :] - 1
        x = node_coords[loc2glb, 0]
        y = node_coords[loc2glb, 1]
        
        [area, gra_1, gra_2] = hat_gradients(x, y)
        
        AK = (np.multiply(gra_1.reshape(-1, 1), gra_1.reshape(1, -1)) + \
              np.multiply(gra_2.reshape(-1, 1), gra_2.reshape(1, -1)))* area
        
        A[np.ix_(loc2glb, loc2glb)] += AK
    
    return A


def mass_assembler_2D(node_coords, element_node_tags):
    number_points    = np.shape(node_coords)[0]
    number_elements  = np.shape(element_node_tags)[0]
    
    # sparse matrix must be used here to optimize !!!
    # sparse matrix to get and storage data is slower :-(: csc_matrix or lil_matrix
    
    M = np.zeros((number_points, number_points))
    
    for i in range(number_elements):
        loc2glb = element_node_tags[i, :] - 1
        x = node_coords[loc2glb, 0]
        y = node_coords[loc2glb, 1]
        
        area = polygon_area(x, y)
        
        MK = np.array([[2,1,1], [1,2,1], [1,1,2]]) / 12 *area
        
        M[np.ix_(loc2glb, loc2glb)] += MK
        
    return M

def f(x, y):
    f = np.sin(x+y)
    #f = 1.0
    return f

def load_assembler_2D(node_coords, element_node_tags, f):
    number_points    = np.shape(node_coords)[0]
    number_elements  = np.shape(element_node_tags)[0]
    
    b = np.zeros((number_points, 1))
    
    for i in range(number_elements):
        loc2glb = element_node_tags[i, :] - 1
        x = node_coords[loc2glb, 0]
        y = node_coords[loc2glb, 1]
        
        area = polygon_area(x, y)
        
        bK = np.array([[f(x[0], y[0])], [f(x[1], y[1])], [f(x[2], y[2])]]) / 3 *area
        
        b[loc2glb] += bK
        
    return b




A = stiffness_assembler_2D(node_coords, element_node_tags)
M = mass_assembler_2D(node_coords, element_node_tags)
b = load_assembler_2D(node_coords, element_node_tags, f)



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
plt.savefig('numpy_cal.jpg', dpi=600)
plt.show()
