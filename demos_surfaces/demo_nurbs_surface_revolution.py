
""" Example showing the creation of an extrusion NURBS surface """


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
from nurbs_curve_circular_arc import NurbsCircularArc
from nurbs_surface_revolution import NurbsSurfaceRevolution


# -------------------------------------------------------------------------------------------------------------------- #
# Revolution surface example
# -------------------------------------------------------------------------------------------------------------------- #

# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.20, 0.00, 0.00]
P[:, 1] = [0.50, 0.00, 0.25]
P[:, 2] = [0.55, 0.00, 0.50]
P[:, 3] = [0.45, 0.00, 0.75]
P[:, 4] = [0.30, 0.00, 1.00]

# Define the array of control point weights
W = np.asarray([1, 2, 3, 2 ,1])

# Create the generatrix NURBS curve
nurbsGeneratrix = NurbsCurve(control_points=P, weights=W)

# Set the a point to define the axis of revolution
axis_point = np.asarray([0.0, 0.0, 0.0])

# Set a direction to define the axis of revolution (needs not be unitary)
axis_direction = np.asarray([0.2, -0.2, 1.0])

# Set the revolution angle
theta_start, theta_end = 0.00, 2*np.pi

# Create and plot the NURBS surface
nurbsRevolutionSurface = NurbsSurfaceRevolution(nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end).NurbsSurface
nurbsRevolutionSurface.plot(surface=True, control_points=True, boundary=True, normals=False)



# -------------------------------------------------------------------------------------------------------------------- #
# Revolution surface example: A sphere
# -------------------------------------------------------------------------------------------------------------------- #

# Create the generatrix NURBS curve
O = np.asarray([0.00, 0.00, 0.00])    # Circle center
X = np.asarray([1.00, 0.00, 0.00])   # Abscissa direction (negative to see normals pointing outwards)
Y = np.asarray([0.00, 0.00, 1.00])    # Ordinate direction
R = 0.5
theta_start, theta_end = np.pi/2, 3/2*np.pi
nurbsGeneratrix = NurbsCircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

# Set the a point to define the axis of revolution
axis_point = np.asarray([0.0, 0.0, 0.0])

# Set a direction to define the axis of revolution (needs not be unitary)
axis_direction = np.asarray([0.0, 0.0, 1.0])

# Set the revolution angle
theta_start, theta_end = np.pi/2, 2*np.pi

# Create and plot the NURBS surface
nurbsRevolutionSurface = NurbsSurfaceRevolution(nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end).NurbsSurface
fig, ax = nurbsRevolutionSurface.plot(surface=True, control_points=False, boundary=False, normals=False)

# Plot isoparametric curves
nurbsRevolutionSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 10), linewidth=0.75)
nurbsRevolutionSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 10), linewidth=0.75)

# Validate the computation of the mean and gaussian curvatures
Nu, Nv, h = 50, 50, 1e-6
u = np.linspace(0+h, 1-h, Nu)      # Small offset to avoid the poles
v = np.linspace(0+h, 1-h, Nv)      # Small offset to avoid the poles
[uu, vv] = np.meshgrid(u, v, indexing='ij')
u, v = uu.flatten(), vv.flatten()
K_mean, K_gauss = nurbsRevolutionSurface.get_curvature(u, v)
error_mean = np.sum(1/(Nu*Nv) * (K_mean - 1/R)**2)**(1/2)
error_gauss = np.sum(1/(Nu*Nv) * (K_gauss - 1/R**2)**2)**(1/2)
print('The two-norm error of the mean curvature computation is      :  ', error_mean)
print('The two-norm error of the gaussian curvature computation is  :  ', error_gauss)


# -------------------------------------------------------------------------------------------------------------------- #
# Revolution surface example: A torus
# -------------------------------------------------------------------------------------------------------------------- #

# Create the generatrix NURBS curve
O = np.asarray([1.00, 0.00, 0.00])    # Circle center
X = np.asarray([1.00, 0.00, 0.00])    # Abscissa direction
Y = np.asarray([0.00, 0.00, 1.00])    # Ordinate direction
R = 0.5
theta_start, theta_end = 0, 2*np.pi
nurbsGeneratrix = NurbsCircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

# Set the a point to define the axis of revolution
axis_point = np.asarray([0.0, 0.0, 0.0])

# Set a direction to define the axis of revolution (needs not be unitary)
axis_direction = np.asarray([0.0, 0.0, 1.0])

# Set the revolution angle
theta_start, theta_end = 0.00, 2*np.pi

# Create and plot the NURBS surface
nurbsRevolutionSurface = NurbsSurfaceRevolution(nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end).NurbsSurface
nurbsRevolutionSurface.plot(surface=True, surface_color='gaussian_curvature', colorbar=True,
                            control_points=False, boundary=True, normals=False)

# Show the figures
plt.show()



