import numpy as np
import matplotlib.pyplot as plt

cov = np.array([[10, 1], [1, 2]])
num_points = 200
initial_points = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=num_points)

eigenvalues, eigenvectors = np.linalg.eig(cov)
order = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

def pca_pdf(x, y, eigenvectors, eigenvalues):
    point = np.array([x, y])
    projection = np.dot(eigenvectors.T, point)
    scaling_factor = np.exp(-0.5 * np.sum(projection**2 / eigenvalues))
    return scaling_factor

angles = np.linspace(0, 2*np.pi, 200)
radii = np.linspace(0, 6, 200)
angles_grid, radii_grid = np.meshgrid(angles, radii)

pdf_values = np.zeros_like(angles_grid)
for i in range(angles_grid.shape[0]):
    for j in range(angles_grid.shape[1]):
        angle = angles_grid[i, j]
        radius = radii_grid[i, j]
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pdf_values[i, j] = pca_pdf(x, y, eigenvectors, eigenvalues)

plt.figure(figsize=(8, 6))
plt.scatter(initial_points[:, 0], initial_points[:, 1], alpha=0.5, label='Data Points')
plt.quiver(0, 0, eigenvectors[0, 0] * 3, eigenvectors[1, 0] * 3, color='r', label='Primary PCA')
plt.quiver(0, 0, eigenvectors[0, 1] * 1.5, eigenvectors[1, 1] * 1.5, color='g', label='Secondary PCA')
X = radii_grid * np.cos(angles_grid)
Y = radii_grid * np.sin(angles_grid)
plt.contour(X, Y, pdf_values, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Data, PCA Directions, and Probability Density')
plt.show()
