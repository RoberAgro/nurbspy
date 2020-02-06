#!/usr/bin/python3

""" Example showing how to create a B-Spline surface """


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
# B-Spline surface example
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

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1
m = np.shape(P)[2] - 1

# Define the order of the basis polynomials
# Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
# Set p = n (number of control points minus one) to obtain a Bezier
p = 3
q = 3

# Define the knot vectors (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
# q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

# Create and plot the B-Spline surface
bsplineSurface = NurbsSurface(control_points=P, u_degree=p, v_degree=q, u_knots=U, v_knots=V)
fig, ax = bsplineSurface.plot(surface=False, control_points=False)

# Plot the blade surface using a colormap based on curvature
bsplineSurface.plot_surface(fig, ax, surface_color='mean_curvature', colorbar=True)

# Plot the vectors normal to the surface
bsplineSurface.plot_normals(fig, ax)

# Plot isoparametric curves
bsplineSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bsplineSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Show the figure
plt.show()
