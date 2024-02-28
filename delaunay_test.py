import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
# 2D Points
points_2d = np.random.rand(20, 2)  # 20 random points in 2D

# 3D Points
points_3d = np.random.rand(5, 3)  # 30 random points in 3D
# 2D Triangulation
tri_2d = Delaunay(points_2d)

# 3D Triangulation
tri_3d = Delaunay(points_3d)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 'o')

# Plot each tetrahedron (a bit more complex)
for simplex in tri_3d.simplices:
    x = np.array([points_3d[simplex[0], 0], points_3d[simplex[1], 0], points_3d[simplex[2], 0], points_3d[simplex[3], 0], points_3d[simplex[0], 0]])
    y = np.array([points_3d[simplex[0], 1], points_3d[simplex[1], 1], points_3d[simplex[2], 1], points_3d[simplex[3], 1], points_3d[simplex[0], 1]])
    z = np.array([points_3d[simplex[0], 2], points_3d[simplex[1], 2], points_3d[simplex[2], 2], points_3d[simplex[3], 2], points_3d[simplex[0], 2]])
    ax.plot_trisurf(x, y, z, color='lightgray', edgecolor='k')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Delaunay Triangulation')
plt.show()