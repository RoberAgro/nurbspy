# nurbspy


## Introduction
`nurbspy` is a lightweight CAD kernel to represent Non-Uniform Rational Basis Splines (NURBS) curves and surfaces. The library is written in Python3 and it relies on the Numpy and Scipy packages to perform the computations 

`nurbspy` can handle real and complex numbers alllowing to compute the derivatives of the coordinates (shape sensitivity) using the complex step method (this information is necessary for gradient-based shape optimization). To our knowledge, this is the only NURBS library that provides the flexibility to work with complex numbers right away.

The library is intended for those who need a simple NURBS library for engineering, teaching and also those who just want to explore the NURBS world without having to program everything from scratch. `nurbspy` does not intend to be a industrial grade CAD kernel (check out OpenCascade if you need a powerful, open source CAD kernel) but it provides a simple alternative for those who do not need a more powerful and complex library.


The implementation of the source code was inspired by the algorithms presented in the NURBS book (reference here) adapting them to exploit vectorized operations on Numpy and Numba JIT compiler to achieve C-like speeds The library contains many references to the equations and algorithms of the NURBS book to guide users who are interested to understand the implementation in detail. 


This library was developed by Roberto Agromayor, PhD candidate at the Norwegian University of Science of Technology, as an outcome of his work on turbomachinery shape optimization.


 
## Dependencies

- Python 3.X or higher
- Numpy X.X or higher
- Scipy X.X or higher
- Matplotlib X.X or higher


## Capabilities

### NURBS curves

The class `NurbsCurve` implements methods to:

- Compute the coordinates of NURBS curves (any number of dimensions)
- Compute analytic derivatives of arbitrary order (any number of dimensions)
- Compute the tangent, notmal and binormal unitary vectors analytically in 2D or 3D (Frenet-Serret frame of reference).
- Compute the curvature and torsion analytically in 2D or 3D
- Compute the arc length of the curve using numerical quadrature
- Routines for 1D, 2D, and 3D plots


#### Example of use

NURBSpy can  be used to create Bézier, B-Spline and NURBS curves. The type of curve depends on the number of arguments used to initialize the class.

For instance, the following code snipped can be used to generate a degree four Bézier curve in three dimensions

```py
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Define the array of control points
P = np.zeros((3,5))
P[:, 0] = [0.20, 0.50, 0.00]
P[:, 1] = [0.40, 0.70, 0.25]
P[:, 2] = [0.80, 0.60, 0.50]
P[:, 3] = [0.80, 0.40, 0.75]
P[:, 4] = [0.40, 0.20, 1.00]

# Create and plot the Bezier curve
bezierCurve = nrb.NurbsCurve(control_points=P)
bezierCurve.plot()
plt.show()
```

See the directory `demos_curves/` to see more examples showing the capabilities of the library.



### NURBS surfaces


The class `NurbsSurface` implements methods to:

- Compute the coordinates of NURBS curves (any number of dimensions)
- Compute analytic derivatives of arbitrary order (any number of dimensions)
- Compute the unitary vector normal to the surface (3D).
- Compute the mean and Gaussian curvatures of the surface (3D)
- Compute u- and v-isoparametic curves (these curves are instances of the NurbsCurve class)
- Routines for 3D plots

In addition, `nurbspy` offers constructors to define some special NURBS surfaces including:

- Bilinear surfaces
- Ruled surfaces
- Extrusion surfaces (general cylinders)
- Revolution surfaces
- Coons surfaces (transfinite patches)



#### Examples of use

The following code snipped shows how to generate a simple NURBS surface

```py
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

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

# Create and plot the Bezier surface
bezierSurface = NurbsSurface(control_points=P)
nurbsSurface.plot()
plt.show()
```

See the directory `demos_surfaces/` to see more examples showing the capabilities of the library and how to use them


