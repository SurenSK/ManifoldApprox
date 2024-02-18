from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from main import display, load_groundTruth

groundTruth = load_groundTruth('2d_manifold.png')
display(groundTruth)