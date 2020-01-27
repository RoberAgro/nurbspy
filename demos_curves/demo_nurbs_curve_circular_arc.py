#!/usr/bin/python3

""" Example showing how to represent circular arcs using NURBS curves """


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


# -------------------------------------------------------------------------------------------------------------------- #
# 2D Circular arc example
# -------------------------------------------------------------------------------------------------------------------- #
# Set defining parameters
O = np.asarray([0.00, 1.00])            # Circle center
X = np.asarray([1.00, 0.00])            # Abscissa direction
Y = np.asarray([0.00, 1.00])            # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 1/6*np.pi                 # Start angle
theta_end = 3/2*np.pi - 1/6*np.pi       # End angle

# Create and plot the circular arc
my_circular_arc = NurbsCircularArc(O, X, Y , R, theta_start, theta_end)
my_circular_arc.plot()

# Check the curvature computation
u = np.linspace(0, 1, 50)
curvature = my_circular_arc.NurbsCurve.get_curvature(u)
print("\n2D circle:")
print("The two-norm of the curvature error is  :  ", np.sum((curvature-1/R)**2)**(1/2))

# Check the arc length computation
arc_length = my_circular_arc.NurbsCurve.get_arclength()
print("The arc length computation error is     :  ", arc_length-R*np.abs(theta_end-theta_start))


# -------------------------------------------------------------------------------------------------------------------- #
# 3D Circular arc example
# -------------------------------------------------------------------------------------------------------------------- #
# Set defining parameters
O = np.asarray([0.00, 0.00, 0.50])      # Circle center
X = np.asarray([3.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.5                                 # Circle radius
theta_start = 1/6*np.pi                 # Start angle
theta_end = np.pi                       # End angle

# Create and plot the circular arc
my_circular_arc = NurbsCircularArc(O, X, Y , R, theta_start, theta_end)
my_circular_arc.plot()

# Check the curvature computation
u = np.linspace(0, 1, 50)
curvature = my_circular_arc.NurbsCurve.get_curvature(u)
print("\n3D circle:")
print("The two-norm of the curvature error is  :  ", np.sum((curvature-1/R)**2)**(1/2))

# Check the arc length computation
arc_length = my_circular_arc.NurbsCurve.get_arclength()
print("The arc length computation error is     :  ", arc_length-R*np.abs(theta_end-theta_start))

# Show the figures
plt.show()

