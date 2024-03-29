from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class P:
    groundTruth = None
    def __init__(self, x, y):
        if P.groundTruth is None:
            raise Exception("P.groundTruth is not set")
        self.x = x
        self.y = y
        self.l = P.groundTruth[x][y]
    def __lt__(self, other):
        return self.l > other.l
    def __repr__(self):
        return f"Point[({self.x:.2f}, {self.y:.2f}):{self.l:.2f}]"
    
def display(image, pts=None):
    image = np.clip(image, 0, 255).astype(np.uint8)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = np.stack((image,) * 3, axis=-1)

    if pts is not None:
        for x, y in pts:
            image[y, x] = [255, 0, 0]

    plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_groundTruth(image_path):
    arr = Image.open(image_path).convert('L')
    np_arr = np.array(arr, dtype=float)
    blur_radius = 2
    np_arr = gaussian_filter(np_arr, sigma=blur_radius)
    mean, std_dev = 0, 8
    np_arr += np.random.normal(mean, std_dev, np_arr.shape)
    np_arr = np.clip(np_arr, 0, 255)
    return np_arr

if __name__ == "__main__":
    groundTruth = load_groundTruth('2d_manifold.png')
    initPoint = [300, 300]
    manifold_points = [initPoint]
    display(groundTruth, manifold_points)
    print("Finished MCMC. Number of points in the manifold: ", len(manifold_points))