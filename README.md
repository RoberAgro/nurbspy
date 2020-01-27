# NURBSpy



### Introduction
NURBSpy is a lightweight CAD kernel to represent NURBS (Non-Uniform Rational Basis Spline) curves and surfaces. The library is written in Python 3.x and it relies on Numpy and Scipy to perform the computations and on Matplotlib for visualization.

NURBSpy can handle real and complex numbers alllowing to compute the derivatives of the coordinates (shape sensitivity) using the complex step method (this information is necessary for gradient-based shape optimization). To our knowledge, this is the only NURBS library that provides the flexibility to work with complex numbers right away.

The library is intended for those who need a simple NURBS library for engineering, teaching and also those who just want to explore the NURBS world without having to program everything from scratch. NURBSpy does not intend to be a industrial grade CAD kernel (check out OpenCascade if you need a powerful, open source CAD kernel) but it provides a simple alternative for those who do not need a more powerful and complex library.


The implementation of the source code was inspired by the algorithms presented in the NURBS book (reference here) adapting them to exploit vectorized operations on Numpy. The library contains many references to the equations and algorithms of the NURBS book to guide users who are interested to understand the implementation in detail. 


This library was developed by Roberto Agromayor, PhD candidate at the Norwegian University of Science of Technology, as an outcome of his work on turbomachinery shape optimization.


 
### Requirements

- Python 3.X or higher
- Numpy X.X or higher
- Scipy X.X or higher
- Matplotlib X.X or higher


### NURBS curves


#### Capabilities
The class `NurbsCurve` implements methods to:

- Compute the coordinates of NURBS curves (any number of dimensions)
- Compute analytic derivatives of arbitrary order (any number of dimensions)
- Compute the tangent, notmal and binormal unitary vectors analytically in 2D or 3D (Frenet-Serret frame of reference).
- Compute the curvature and torsion analytically in 2D or 3D
- Compute the arc length of the curve using numerical quadrature
- Routines for 1D, 2D, and 3D plots


#### Examples of use

NURBSpy can  be used to create Bézier, B-Spline and NURBS curves. The type of curve depends on the number of arguments used to initialize the class.

For instance, the following code snipped can be used to generate a fourth degree polynomial Bézier curve in 2D

```py
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from nurbs_curve import NurbsCurve

# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.60, 0.20]
P[:, 4] = [0.40, 0.20]

# Create and plot the Bezier curve
bezier2D = NurbsCurve(control_points=P)
bezier2D.plot_curve()
plt.show()

```

Whereas the following code snipped can be used to generate a third degree rational Bézier curve in 3D

```py
# Import packages
import numpy as np
from nurbs_curve import NurbsCurve

# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.20, 0.50, 0.00]
P[:, 1] = [0.40, 0.70, 0.25]
P[:, 2] = [0.80, 0.60, 0.50]
P[:, 3] = [0.80, 0.40, 0.75]
P[:, 4] = [0.40, 0.20, 1.00]

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1

# Define the array of control point weights
W = np.asarray([1, 2, 3, 2, 1])

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create and plot the B-Spline curve
nurbs3D = NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
nurbs3D.plot_curve()
plt.show()
```

Note the both curves are generated as instances of the NurbsCurve class. For the case of the 3D NURBS curve the control points and weights, degree and knot vector are provided at construction, whereas for the case of the 2D Bézier curve the only argument is the array of control points and the weghts, curve degree, and knot sequence are generated internally.

See the directory `demos_curves/` to see more examples showing the capabilities of the library and how to use them



### NURBS surfaces

#### Capabilities

The class `NurbsSurface` implements methods to:

- Compute the coordinates of NURBS curves (any number of dimensions)
- Compute analytic derivatives of arbitrary order (any number of dimensions)
- Compute the unitary vector normal to the surface (3D).
- Compute the mean and Gaussian curvatures of the surface (3D)
- Compute u- and v-isoparametic curves (these curves are instances of the NurbsCurve class)
- Routines for 3D plots

In addition, NURBSpy contains classes to define special types of NURBS surfaces including:

- Bilinear surfaces
- Ruled surfaces
- Extrusion surfaces (general cylinders)
- Revolution surfaces
- Coons surfaces (transfinite surfaces)



#### Examples of use

The following code snipped shows how to generate a simple NURBS surface

```py
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from nurbs_surface import NurbsSurface

# Define the array of control points
n_dim, n, m = 3, 3, 2
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 3.00, 0.00]
P[:, 1, 0] = [1.00, 2.00, 0.00]
P[:, 2, 0] = [2.00, 1.50, 0.00]

# Second row
P[:, 0, 1] = [0.00, 3.00, 1.00]
P[:, 1, 1] = [1.00, 2.00, 1.00]
P[:, 2, 1] = [2.00, 1.50, 1.00]

# Define the array of control point weights
W = np.zeros((n, m))
W[:, 0] = np.asarray([1, 1, 1])		# First row
W[:, 1] = np.asarray([1, 1, 1])		# Second row

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1
m = np.shape(P)[2] - 1

# Define the order of the basis polynomials
# Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
# Set p = n (number of control points minus one) to obtain a Bezier
p = 2
q = 1

# Define the knot vectors (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
# q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

# Create and plot the NURBS surface
nurbsSurface = NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)
nurbsSurface.plot_surface()
plt.show()
```

The next example shows how to define a ruled surface using the NurbsSurfaceRuled() class


```py
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from nurbs_curve import NurbsCurve
from nurbs_surface_ruled import NurbsSurfaceRuled

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

# Show the figure
plt.show()
```


See the directory `demos_surfaces/` to see more examples showing the capabilities of the library and how to use them


