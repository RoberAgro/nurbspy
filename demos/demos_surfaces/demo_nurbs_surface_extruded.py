#!/usr/bin/python3

""" Example showing how to create an extruded NURBS surface (general cylinder) """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Extruded surface example:
# -------------------------------------------------------------------------------------------------------------------- #
# Define the base NURBS curve (rational Bézier curve)
P = np.zeros((3, 5))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.50, 0.80, 0.00]
P[:, 2] = [0.75, 0.60, 0.00]
P[:, 3] = [1.00, 0.30, 0.00]
P[:, 4] = [0.80, 0.10, 0.00]
W = np.asarray([1, 1, 1, 1, 1])
nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W)

# Set the extrusion direction (can be unitary or not)
direction = np.asarray([1, 1, 5])

# Set the extrusion length
length = 1

# Create and plot the ruled NURBS surface
extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface
fig, ax = extrudedNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
extrudedNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 5))
extrudedNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the base NURBS curve
nurbsCurve.plot_curve(fig, ax, color='b', linewidth=2.5)



# -------------------------------------------------------------------------------------------------------------------- #
# Extruded surface example: A circular cylinder
# -------------------------------------------------------------------------------------------------------------------- #

# Define a circular arc
O = np.asarray([0.00, 0.00, 0.00])      # Circle center
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 0.00                      # Start angle
theta_end = 2*np.pi                     # End angle

# Create and plot the circular arc
nurbsCurve = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Set the extrusion direction (can be unitary or not)
direction = np.asarray([0, 0, 1])

# Set the extrusion length
length = 1

# Create and plot the ruled NURBS surface
extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface
fig, ax = extrudedNurbsSurface.plot(surface=True, surface_color='blue', control_points=True)

# Plot isoparametric curves
extrudedNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 9))
extrudedNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))

# Plot the base NURBS curve
nurbsCurve.plot_curve(fig, ax, color='b', linewidth=2.5)

# Show the figure
plt.show()


