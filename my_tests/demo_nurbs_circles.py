#!/usr/bin/python3

""" Example showing how to parametrize a circle using NURBS curves """


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


# -------------------------------------------------------------------------------------------------------------------- #
# Create a circle using 4 arcs of 90 degrees (NURBS book section 7.5)
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
P = np.zeros((2,9))
P[:, 0] = [ 1.00,  0.00]
P[:, 1] = [ 1.00,  1.00]
P[:, 2] = [ 0.00,  1.00]
P[:, 3] = [-1.00,  1.00]
P[:, 4] = [-1.00,  0.00]
P[:, 5] = [-1.00, -1.00]
P[:, 6] = [ 0.00, -1.00]
P[:, 7] = [ 1.00, -1.00]
P[:, 8] = [ 1.00,  0.00]

# Define the array of control point weights
W = np.asarray([1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1])

# Define the order of the basis polynomials
p = 2

# Define the knot vector (clamped spline)
# p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
U = np.asarray([0, 0, 0, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 1, 1, 1])

# Create the NURBS curve
circle1 = NurbsCurve(P, W, p, U)


# -------------------------------------------------------------------------------------------------------------------- #
# Create a circle using 3 arcs of 120 degrees (NURBS book section 7.5)
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
a = np.cos(np.pi/6)
P = np.zeros((2,7))
P[:, 0] = [   a, 1/2]
P[:, 1] = [   0,   2]
P[:, 2] = [  -a, 1/2]
P[:, 3] = [-2*a,  -1]
P[:, 4] = [   0,  -1]
P[:, 5] = [ 2*a,  -1]
P[:, 6] = [   a, 1/2]

# Define the array of control point weights
W = np.asarray([1, 1/2, 1, 1/2, 1, 1/2, 1])

# Define the order of the basis polynomials
p = 2

# Define the knot vector (clamped spline)
# p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
U = np.asarray([0, 0, 0, 1/3, 1/3, 2/3, 2/3, 1, 1, 1])

# Create the NURBS curve
circle2 = NurbsCurve(P, W, p, U)


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the NURBS curve
# -------------------------------------------------------------------------------------------------------------------- #
circle1.plot_curve(curve='yes', control_points='yes')
circle2.plot_curve(curve='yes', control_points='yes')
plt.show()

