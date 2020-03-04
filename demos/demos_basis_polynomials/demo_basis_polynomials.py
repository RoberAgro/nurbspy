#!/usr/bin/python3

""" Example showing how to compute a family of basis polynomials """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy as nbp
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Basis polynomials and derivatives example
# -------------------------------------------------------------------------------------------------------------------- #
# Maximum index of the basis polynomials (counting from zero)
n = 4

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Define a new u-parametrization suitable for finite differences
h = 1e-5
hh = h + h**2
Nu = 1000
u = np.linspace(0.00 + hh, 1.00 - hh, Nu)       # Make sure that the limits [0, 1] also work when making changes

# Compute the basis polynomials and derivatives
N_basis   = nbp.compute_basis_polynomials(n, p, U, u)
dN_basis  = nbp.compute_basis_polynomials_derivatives(n, p, U, u, derivative_order=1)
ddN_basis = nbp.compute_basis_polynomials_derivatives(n, p, U, u, derivative_order=2)


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
# Create the figure
fig = plt.figure(figsize=(15, 5))

# Plot the basis polynomials
ax1 = fig.add_subplot(131)
ax1.set_title('Zeroth derivative', fontsize=12, color='k', pad=12)
ax1.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax1.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax1.plot(u, N_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Plot the first derivative
ax2 = fig.add_subplot(132)
ax2.set_title('First derivative', fontsize=12, color='k', pad=12)
ax2.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax2.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax2.plot(u, dN_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Plot the second derivative
ax3 = fig.add_subplot(133)
ax3.set_title('Second derivative', fontsize=12, color='k', pad=12)
ax3.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax3.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax3.plot(u, ddN_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Create legend
ax3.legend(ncol=1, loc='right', bbox_to_anchor=(1.60, 0.50), fontsize=10, edgecolor='k', framealpha=1.0)

# Adjust pad
plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

# Show the figure
plt.show()



# # -------------------------------------------------------------------------------------------------------------------- #
# # Check that the computations are correct
# # -------------------------------------------------------------------------------------------------------------------- #
# # Check that the sum of the basis polynomials is equal to one (partition of unity property)
# print('The two-norm of partition of unity error is     :  ', np.sum((np.sum(N_basis, axis=0) - 1.00) ** 2) ** (1 / 2))
#
# # Check the first derivative against a finite difference aproximation
# a = -1/2*compute_basis_polynomials(n, p, U, u - h)
# b = +1/2*compute_basis_polynomials(n, p, U, u + h)
# dN_fd = (a+b)/h
# print('The two-norm of the first derivative error is   :  ', np.sum((dN_basis-dN_fd)**2)**(1/2)/Nu)
#
# # Check the second derivative against a finite difference aproximation
# a = +1*compute_basis_polynomials(n, p, U, u - h)
# b = -2*compute_basis_polynomials(n, p, U, u)
# c = +1*compute_basis_polynomials(n, p, U, u + h)
# ddN_fd = (a+b+c)/h**2
# print('The two-norm of the second derivative error is  :  ', np.sum((ddN_basis-ddN_fd)**2)**(1/2)/Nu)


