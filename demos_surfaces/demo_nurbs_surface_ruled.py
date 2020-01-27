#!/usr/bin/python3

""" Example showing how to create a ruled NURBS surface """


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
from nurbs_curve import NurbsCurve
from nurbs_surface import NurbsSurface
from nurbs_surface_ruled import NurbsSurfaceRuled
from nurbs_curve_circular_arc import NurbsCircularArc


# -------------------------------------------------------------------------------------------------------------------- #
# Ruled surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Define the lower NURBS curve (rational Bézier curve)
P1 = np.zeros((3, 5))
P1[:, 0] = [0.00, 0.00, 0.00]
P1[:, 1] = [0.25, 0.00, 0.50]
P1[:, 2] = [0.50, 0.00, 0.50]
P1[:, 3] = [0.75, 0.00, 0.00]
P1[:, 4] = [1.00, 0.00, 0.00]
W1 = np.asarray([1, 1, 2, 1, 1])
nurbsCurve1 = NurbsCurve(control_points=P1, weights=W1)

# Define the lower NURBS curve (rational Bézier curve)
P2 = np.zeros((3, 5))
P2[:, 0] = [0.00, 1.00, 0.50]
P2[:, 1] = [0.25, 1.00, 0.00]
P2[:, 2] = [0.50, 1.00, 0.00]
P2[:, 3] = [0.75, 1.00, 0.50]
P2[:, 4] = [1.00, 1.00, 0.50]
W2 = np.asarray([1, 1, 2, 1, 1])
nurbsCurve2 = NurbsCurve(control_points=P2, weights=W2)

# Create and plot the ruled NURBS surface
ruledNurbsSurface = NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
fig, ax = ruledNurbsSurface.plot(surface=True, surface_color='red', control_points=True)

# Plot isoparametric curves
ruledNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
ruledNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the upper and lower NURBS curves
nurbsCurve1.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve2.plot_curve(fig, ax, color='g', linewidth=2.5)






# -------------------------------------------------------------------------------------------------------------------- #
# Ruled surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Create the first circular arc
O = np.asarray([0.00, 0.00, 0.00])      # Circle center
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 0.00                      # Start angle
theta_end = 2*np.pi                     # End angle
nurbsCurve1 = NurbsCircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Create the second circular arc
O = np.asarray([0.00, 0.00, 0.00])      # Circle center
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 0.00, 1.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = np.pi/2                   # Start angle
theta_end = 2*np.pi+np.pi/2             # End angle
nurbsCurve2 = NurbsCircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Create and plot the ruled NURBS surface
ruledNurbsSurface = NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
fig, ax = ruledNurbsSurface.plot(surface=False, surface_color='red', control_points=False)

# Plot isoparametric curves
ruledNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 30))
# ruledNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the upper and lower NURBS curves
nurbsCurve1.plot_curve(fig, ax, color='b', linewidth=2.5)
nurbsCurve2.plot_curve(fig, ax, color='g', linewidth=2.5)
plt.show()