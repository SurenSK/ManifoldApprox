from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from main import display, load_groundTruth, P
import heapq

groundTruth = load_groundTruth('2d_manifold.png')
# P.groundTruth = groundTruth
import emcee
import numpy as np
from PIL import Image

image_path = '2d_manifold.png'
image = Image.open(image_path).convert('L')
image_data = np.asarray(image)
image_data = image_data / 255.0
def lnprob(xy, img_data):
    x, y = int(xy[0]), int(xy[1])
    if x < 0 or y < 0 or x >= img_data.shape[1] or y >= img_data.shape[0]:
        return -np.inf
    return np.log(img_data[y, x] + 1e-5)

burnin = 1000
ndim = 2
nwalkers = 100
nsteps = 10000
p0 = np.random.rand(nwalkers, ndim) * np.array(image_data.shape[::-1])

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[image_data])
sampler.run_mcmc(p0, nsteps)
samples = sampler.get_chain(discard=burnin, flat=True)
manifold_points = []
threshold = 0.5
for x, y in samples:
    x, y = int(x), int(y)
    if image_data[y, x] > threshold:
        manifold_points.append((x, y))
display(groundTruth, manifold_points)