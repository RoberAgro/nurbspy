#!/usr/bin/python3

""" Example showing how to create a rational Bezier surface """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing general packages
# -------------------------------------------------------------------------------------------------------------------- #
import sys
import os
import time
import pdb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Importing user-defined packages
# -------------------------------------------------------------------------------------------------------------------- #
sys.path.append(os.getcwd() + '/../src/')
from nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Rational Bezier surface example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
n_dim, n, m = 3, 5, 4
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 3.00, 0.00]
P[:, 1, 0] = [1.00, 2.00, 0.00]
P[:, 2, 0] = [2.00, 1.50, 0.00]
P[:, 3, 0] = [3.00, 2.00, 0.00]
P[:, 4, 0] = [4.00, 3.00, 0.00]

# Second row
P[:, 0, 1] = [0.00, 3.00, 1.00]
P[:, 1, 1] = [1.00, 2.00, 1.00]
P[:, 2, 1] = [2.00, 1.50, 1.00]
P[:, 3, 1] = [3.00, 2.00, 1.00]
P[:, 4, 1] = [4.00, 3.00, 1.00]

# Third row
P[:, 0, 2] = [0.00, 3.00, 2.00]
P[:, 1, 2] = [1.00, 2.00, 2.00]
P[:, 2, 2] = [2.00, 1.50, 2.00]
P[:, 3, 2] = [3.00, 2.00, 2.00]
P[:, 4, 2] = [4.00, 3.00, 2.00]

# Fourth row
P[:, 0, 3] = [0.50, 3.00, 3.00]
P[:, 1, 3] = [1.00, 2.50, 3.00]
P[:, 2, 3] = [2.00, 2.00, 3.00]
P[:, 3, 3] = [3.00, 2.50, 3.00]
P[:, 4, 3] = [3.50, 3.00, 3.00]

# Define the array of control point weights
W = np.zeros((n, m))
W[:, 0] = np.asarray([1, 1, 5, 1, 1])
W[:, 1] = np.asarray([1, 1, 5, 1, 1])
W[:, 2] = np.asarray([1, 1, 5, 1, 1])
W[:, 3] = np.asarray([1, 1, 0.5, 1, 1])

# Create and plot the Bezier surface
bezierSurface = NurbsSurface(control_points=P, weights=W)
bezierSurface.plot()
plt.show()
