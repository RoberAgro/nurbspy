# nurbspy


## Description
`nurbspy` is a lightweight, object-oriented library for Non-Uniform Rational Basis Spline (NURBS) curves and surfaces implemented in Python. 
The library was inspired by the equations and algorithms presented in [The NURBS Book](https://doi.org/10.1007/978-3-642-59223-2) and it was implemented using vectorized [Numpy](https://numpy.org/) functions and [Numba's](http://numba.pydata.org/) just-in-time compilation decorators to achieve C-like speed.
 
 Note that `nurbspy` aims to be a easy-to-install and easy-to-use NURBS library, but not a fully fledged CAD kernel. You can check out [OpenCascade](https://www.opencascade.com/doc/occt-7.4.0/overview/html/index.html) if you need a powerful, open source CAD kernel.



## Capabilities



`nurbspy` has the following features to work with NURBS curves:

- Constructors for rational and non-rational Bézier and B-Spline curves
- Methods to evaluate curve coordinates
- Methods to evaluate arbitrary-order derivatives analytically
- Methods to evaluate the tangent, normal, and binormal unitary vectors (Frenet-Serret frame of reference)
- Methods to compute the curvature and torsion
- Methods to compute the arc-length of the curve by numerical quadrature
- Methods to visualize the curve using the Matplotlib library


In addition, `nurbspy` provides the following capabilities to work with NURBS surfaces:


- Constructors for rational and non-rational Bézier and B-Spline surfaces
- Additional constructors for some common special surfaces:
	- Bilinear surfaces
	- Ruled surfaces
	- Extruded surfaces
	- Revolution surfaces
	- Coons surfaces
- Methods to evaluate surface coordinates
- Methods to evaluate arbitrary-order derivatives analytically
- Methods to evaluate the unitary normal vector
- Methods to evaluate the mean and Gaussian curvatures
- Methods to compute u- and v-isoparametic curves
- Methods to visualize the surface using the Matplotlib library


In addition,  `nurbspy` can work with real and complex data types natively. This allows to compute accurate shape derivatives using the complex step method and avoid the numerical error incurred by finite-difference derivative approximations. Shape sensitivity information is necessary to solve shape optimization problems with many design variables using gradient based-optimization algorithms. To our knowledge, `nurbspy` is the only Python package that provides the flexibility to work with complex numbers right away.



 
## Installation


`nurbspy` has the following mandatory runtime dependencies:

 - `numpy` (multidimensional array library)
 - `scipy` (scientific computing library)
 - `numba` (just-in-time Python compiler)
 - `matplotlib` (visualization library)
 
In addition `nurbspy` uses `pytest` for local tests.


`nurbspy` is available on Linux via the [pip](https://pip.pypa.io/en/stable/) package manager. The installation with pip is straightfoward:

	pip install nurbspy


`nurbspy` is also available on Linux via the [conda](https://pip.pypa.io/en/stable/) package manager thanks to the infrastructure provided by [conda-forge](https://conda-forge.org/). In order to install `nurbspy` via conda you need to add `conda-forge` to your channels and then use the install command

	conda config --add channels conda-forge
	conda install nurbspy



You can verify that `nurbspy` was successfully installed by running the provided below.


## Minimum working examples

### NURBS curves

NURBSpy can  be used to create Bézier, B-Spline and NURBS curves. The type of curve depends on the number of arguments used to initialize the class. As an example, the following code snippet can be used to generate a degree four Bézier curve in two dimensions

```py
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Define the array of control points
P = np.zeros((2,5))
P[:, 0] = [0.20, 0.50]
P[:, 1] = [0.40, 0.70]
P[:, 2] = [0.80, 0.60]
P[:, 3] = [0.80, 0.40]
P[:, 4] = [0.40, 0.20]

# Create and plot the Bezier curve
bezierCurve = nrb.NurbsCurve(control_points=P)
bezierCurve.plot()
plt.show()
```

If the installation was succesful, you should be able to see the Bézier curve when you execute the previous code snippet.

<p style="margin-bottom:1cm;"> </p>
<p align="center">
        <img src="docs/images/curve_example.pdf" height="300" width="300"/>
</p>
<p style="margin-bottom:1cm;"> </p>


Check out the `demos_curves/` directory to see more examples showing the capabilities of the library and how to use them.


### NURBS surfaces

The following code snippet shows how to generate a simple Bézier patch of degree 2 in the u-direction and degree 1 in the v-direction:

```py
# Import packages
import numpy as np
import nurbspy as nrb
import matplotlib.pyplot as plt

# Define the array of control points
n_dim, n, m = 3, 4, 3
P = np.zeros((n_dim, n, m))

# First row
P[:, 0, 0] = [0.00, 0.00, 0.00]
P[:, 1, 0] = [1.00, 0.00, 1.00]
P[:, 2, 0] = [2.00, 0.00, 1.00]
P[:, 3, 0] = [3.00, 0.00, 0.00]

# Second row
P[:, 0, 1] = [0.00, 1.00, 1.00]
P[:, 1, 1] = [1.00, 1.00, 2.00]
P[:, 2, 1] = [2.00, 1.00, 2.00]
P[:, 3, 1] = [3.00, 1.00, 1.00]

# Third row
P[:, 0, 2] = [0.00, 2.00, 0.00]
P[:, 1, 2] = [1.00, 2.00, 1.00]
P[:, 2, 2] = [2.00, 2.00, 1.00]
P[:, 3, 2] = [3.00, 2.00, 0.00]

# Create and plot the Bezier surface
bezierSurface = nrb.NurbsSurface(control_points=P)
bezierSurface.plot(control_points=True, isocurves_u=6, isocurves_v=6)
plt.show()
```

If the installation was succesful, you should be able to see the Bézier surface when you execute the previous code snippet.

<p style="margin-bottom:1cm;"> </p>
<p align="center">
        <img src="docs/images/suface_example.pdf" height="300" width="300"/>
</p>
<p style="margin-bottom:1cm;"> </p>

Check out the `demos_surfaces/` directory to see more examples showing the capabilities of the library and how to use them.


## Contact information
`nurbspy` was developed by PhD candidate [Roberto Agromayor](https://www.ntnu.edu/employees/roberto.agromayor) under the supervision of Associate Professor [Lars O. Nord](https://www.ntnu.edu/employees/lars.nord) at the [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no/) as part of his work on turbomachinery shape optimization.

Please, drop us an email to [roberto.agromayor@ntnu.no](mailto:roberto.agromayor@ntnu.no) if you have questions about the code or you have a bug to report.