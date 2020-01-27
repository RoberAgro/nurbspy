#!/usr/bin/python3

""" Example showing how to create a bilinear NURBS surface """


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
sys.path.append(os.getcwd() + '/../functions/')
from nurbs_surface import NurbsSurface
from nurbs_surface_bilinear import NurbsSurfaceBilinear


# -------------------------------------------------------------------------------------------------------------------- #
# Bilinear patch example: Plane
# -------------------------------------------------------------------------------------------------------------------- #

# Set the bilinear surface defining points
P00 = np.asarray([0.00, 0.00, 0.00])
P01 = np.asarray([2.00, 0.50, 0.00])
P10 = np.asarray([0.20, 1.00, 0.00])
P11 = np.asarray([1.80, 1.50, 0.00])

# Create and plot the NURBS surface
bilinearNurbsSurface = NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
fig, ax = bilinearNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
bilinearNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bilinearNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))


# -------------------------------------------------------------------------------------------------------------------- #
# Bilinear patch example: Hyperbolic paraboloid
# -------------------------------------------------------------------------------------------------------------------- #

# Set the bilinear surface defining points
P00 = np.asarray([0.00, 0.00, 0.00])
P01 = np.asarray([1.00, 0.00, 1.00])
P10 = np.asarray([0.00, 1.00, 1.00])
P11 = np.asarray([1.00, 1.00, 0.00])

# Create and plot the NURBS surface
bilinearNurbsSurface = NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
fig, ax = bilinearNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
bilinearNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
bilinearNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Show the surface
plt.show()



