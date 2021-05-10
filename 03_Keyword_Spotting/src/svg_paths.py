from svg import *
from skimage import io
import matplotlib.pyplot as plt

p270svg = "../res/ground-truth/locations/270.svg"
p270jpg = "../res/images/270.jpg"

svg = SVG(p270svg)
jpg = io.imread(p270jpg)

plt.imshow(jpg)

print(svg)
