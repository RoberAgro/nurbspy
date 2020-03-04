
""" Example showing how to create an annular NURBS surface as a Coons patch """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Coons surface example: Annulus NURBS surface
# -------------------------------------------------------------------------------------------------------------------- #

# Define the outer circle
O = np.asarray([0.00, 0.00, 0.00])      # Circle center
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 1.00                                # Circle radius
theta_start = 0.00                      # Start angle (set any arbitrary value)
theta_end = 2*np.pi                     # End angle (set any arbitrary value)
nurbsOuter = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Define the inner circle
O = np.asarray([0.00, 0.00, 0.00])      # Circle center
X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
R = 0.50                                # Circle radius
nurbsInner = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

# Define start closing boundary
P = np.zeros((3, 2))
P[:, 0] = nurbsOuter.P[:, 0]
P[:, 1] = nurbsInner.P[:, 0]
W = np.asarray([nurbsOuter.W[0], nurbsInner.W[0]])
nurbsWest = nrb.NurbsCurve(control_points=P, weights=W)

# Define the west boundary (rational Bézier curve)
P = np.zeros((3, 2))
P[:, 0] = nurbsOuter.P[:, -1]
P[:, 1] = nurbsInner.P[:, -1]
W = np.asarray([nurbsOuter.W[-1], nurbsInner.W[-1]])
nurbsEast = nrb.NurbsCurve(control_points=P, weights=W)

# Create and plot the Coons NURBS surface to represent the annulus
coonsNurbsSurface = nrb.NurbsSurfaceCoons(nurbsOuter, nurbsInner, nurbsWest, nurbsEast).NurbsSurface
# coonsNurbsSurface = nrb.NurbsSurfaceRuled(nurbsOuter, nurbsInner).NurbsSurface

# Plot the annulus surface and some isoparametric curve
fig, ax = coonsNurbsSurface.plot(surface=True, surface_color='blue', control_points=False)
coonsNurbsSurface.plot_isocurve_u(fig, ax, np.linspace(0, 1, 20))
coonsNurbsSurface.plot_isocurve_v(fig, ax, np.linspace(0, 1, 5))
plt.show()

