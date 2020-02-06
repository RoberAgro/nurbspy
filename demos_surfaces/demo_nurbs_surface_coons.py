#!/usr/bin/python3

""" Example showing how to create a Coons NURBS surface """


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
from nurbs_curve import NurbsCurve
from nurbs_surface import NurbsSurface
from nurbs_surface_coons import NurbsSurfaceCoons


# -------------------------------------------------------------------------------------------------------------------- #
# Coons surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Define the south boundary (rational Bézier curve)
P = np.zeros((3, 4))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.33, 0.00, -0.40]
P[:, 2] = [0.66, 0.10, 0.60]
P[:, 3] = [1.00, 0.20, 0.40]
W = np.asarray([1, 2, 2, 1])
nurbsCurve_south = NurbsCurve(control_points=P, weights=W)

# Define the north boundary (rational Bézier curve)
P = np.zeros((3, 4))
P[:, 0] = [0.05, 1.00, 0.00]
P[:, 1] = [0.33, 1.15, 0.40]
P[:, 2] = [0.66, 1.15, 0.00]
P[:, 3] = [1.05, 1.25, 0.40]
W = np.asarray([1, 2, 2, 1])
nurbsCurve_north = NurbsCurve(control_points=P, weights=W)


# Define the west boundary (rational Bézier curve)
P = np.zeros((3, 3))
P[:, 0] = nurbsCurve_south.P[:, 0]
P[:, 1] = [-0.20, 0.50, -0.40]
P[:, 2] = nurbsCurve_north.P[:, 0]
W = np.asarray([nurbsCurve_south.W[0], 1, nurbsCurve_north.W[0]])
nurbsCurve_west = NurbsCurve(control_points=P, weights=W)


# Define the east boundary (rational Bézier curve)
P = np.zeros((3, 3))
P[:, 0] = nurbsCurve_south.P[:, -1]
P[:, 1] = [1.15, 0.50, 0.30]
P[:, 2] = nurbsCurve_north.P[:, -1]
W = np.asarray([nurbsCurve_south.W[-1], 1, nurbsCurve_north.W[-1]])
nurbsCurve_east = NurbsCurve(control_points=P, weights=W)


# Create and plot the Coons NURBS surface
coonsNurbsSurface = NurbsSurfaceCoons(nurbsCurve_south, nurbsCurve_north, nurbsCurve_west, nurbsCurve_east).NurbsSurface
fig, ax = coonsNurbsSurface.plot(surface=True, surface_color='blue', control_points=False)


# Plot isoparametric curves
coonsNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
coonsNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))


# Plot the boundary NURBS curves
nurbsCurve_south.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_north.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_west.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve_east.plot_curve(fig, ax, color='b', linewidth=2.5)
plt.show()



