"""
import cv2
from matplotlib.pylab import cm

def colorize(image, colormap):
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    colorized = colormap(im)
    print colorized
    cv2.imshow("colorized", colorized)
    cv2.waitKey(0)
    cv2.imwrite("colorized.jpg", colorized*255)

if __name__ == "__main__":
    import sys
    colorize("dog.jpg", cm.jet)
"""
"""
import cv2
import numpy as np

image = cv2.imread('images.jpg',0)
cv2.imshow('mywin2', image)
cv2.imshow('mywin', cv2.applyColorMap(image,4))
cv2.waitKey(0)
"""


import numpy
import matplotlib.pylab as plt
m = [[0.0, 1.47, 2.43, 3.44, 1.08, 2.83, 1.08, 2.13, 2.11, 3.7], [1.47, 0.0, 1.5,2.39, 2.11, 2.4, 2.11, 1.1, 1.1, 3.21], [2.43, 1.5, 0.0, 1.22, 2.69, 1.33, 3.39, 2.15, 2.12, 1.87], [3.44, 2.39, 1.22, 0.0, 3.45, 2.22, 4.34, 2.54, 3.04, 2.28], [1.08, 2.11, 2.69, 3.45, 0.0, 3.13, 1.76, 2.46, 3.02, 3.85], [2.83, 2.4, 1.33, 2.22, 3.13, 0.0, 3.83, 3.32, 2.73, 0.95], [1.08, 2.11, 3.39, 4.34, 1.76, 3.83, 0.0, 2.47, 2.44, 4.74], [2.13, 1.1, 2.15, 2.54, 2.46, 3.32, 2.47, 0.0, 1.78, 4.01], [2.11, 1.1, 2.12, 3.04, 3.02, 2.73, 2.44, 1.78, 0.0, 3.57], [3.7, 3.21, 1.87, 2.28, 3.85, 0.95, 4.74, 4.01, 3.57, 0.0]]
matrix = numpy.matrix(m)
print matrix
fig=plt.figure("hello",figsize=(matrix.shape[0]+5, matrix.shape[1]))
#plt.subplot(211)
plt.imshow(matrix, interpolation='nearest',cmap=plt.cm.ocean)
plt.title("interactive test")
plt.xlabel("index")
plt.colorbar()
plt.show(block=True)

fig=plt.figure("hello",figsize=(matrix.shape[0]+5, matrix.shape[1]))
#plt.subplot(211)
plt.imshow(matrix, interpolation='nearest',cmap=plt.cm.Blues)
plt.title("interactive test")
plt.xlabel("index")
plt.colorbar()
plt.show(block=True)
"""

import matplotlib.pyplot as plt
plt.figure(1, figsize=(5,10))                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])


plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.show()
"""