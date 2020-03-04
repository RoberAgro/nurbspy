
""" Example showing how to combine several NURBS surfaces into a single NURBS surface
    The NURBS surface must be of the same order to merge them
    Merging curves of different degree would require degree elevation (not implemented yet)

"""


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Create the first NURBS surface
# -------------------------------------------------------------------------------------------------------------------- #

# Define the array of control points
n_dim, n, m = 3, 2, 2
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 0.00, 0.00]
P[:, 1, 0] = [0.00, 1.00, 0.00]
P[:, 1, 0] = [0.00, 1.00, 0.00]

# Second row
P[:, 0, 1] = [1.00, 0.00, 0.00]
P[:, 1, 1] = [1.00, 1.00, 0.00]

# Create and plot the NURBS surface
nurbsSurface1 = nrb.NurbsSurface(control_points=P)


# -------------------------------------------------------------------------------------------------------------------- #
# Create the second NURBS surface
# -------------------------------------------------------------------------------------------------------------------- #

# Define the array of control points
n_dim, n, m = 3, 2, 2
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 1.00, 0.00]
P[:, 1, 0] = [0.00, 2.00, 0.00]

# Second row
P[:, 0, 1] = [1.00, 1.00, 0.00]
P[:, 1, 1] = [1.00, 2.00, 0.00]

# Create and plot the NURBS surface
nurbsSurface2 = nrb.NurbsSurface(control_points=P)


# -------------------------------------------------------------------------------------------------------------------- #
# Create the third NURBS surface
# -------------------------------------------------------------------------------------------------------------------- #

# Define the array of control points
n_dim, n, m = 3, 2, 2
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [1.00, 0.00, 0.00]
P[:, 1, 0] = [1.00, 1.00, 0.00]

# Second row
P[:, 0, 1] = [2.00, 0.00, 0.00]
P[:, 1, 1] = [2.00, 1.00, 0.00]

# Create and plot the NURBS surface
nurbsSurface3 = nrb.NurbsSurface(control_points=P)



# -------------------------------------------------------------------------------------------------------------------- #
# Merge two NURBS surfaces with a common u-side
# -------------------------------------------------------------------------------------------------------------------- #

# Merge the surfaces
nurbsMerged_u = nurbsSurface1.attach_nurbs_udir(nurbsSurface2)
nurbsMerged_u.P[2, :, :] = 0.5

# Plot the merged surfaces
fig, ax = nurbsSurface1.plot(control_points=True)
nurbsSurface2.plot(fig, ax, control_points=True)
nurbsMerged_u.plot(fig, ax, control_points=True, surface_color='red')


# -------------------------------------------------------------------------------------------------------------------- #
# Merge two NURBS surfaces with a common v-side
# -------------------------------------------------------------------------------------------------------------------- #

# Merge the surfaces
nurbsMerged_v = nurbsSurface1.attach_nurbs_vdir(nurbsSurface3)
nurbsMerged_v.P[2, :, :] = 0.5

# Plot the merged surfaces
fig, ax = nurbsSurface1.plot(control_points=True)
nurbsSurface3.plot(fig, ax, control_points=True)
nurbsMerged_v.plot(fig, ax, control_points=True, surface_color='red')

# Show the figures
plt.show()

