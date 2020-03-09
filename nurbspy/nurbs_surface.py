# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import scipy.special
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits  import mplot3d
from .nurbs_curve  import NurbsCurve
from .nurbs_basis_functions  import compute_basis_polynomials, compute_basis_polynomials_derivatives


# -------------------------------------------------------------------------------------------------------------------- #
# Define the NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurface:

    """ Create a NURBS (Non-Uniform Rational Basis Spline) surface object

        Parameters
        ----------
        control_points : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        weights : ndarray with shape (n+1, m+1)
            Array containing the weight of the control points
            The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
            The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

        u_degree : int
            Degree of the u-basis polynomials

        v_degree : int
            Degree of the v-basis polynomials

        u_knots : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        v_knots : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        Notes
        -----
        This class includes methods to compute:

            - Surface coordinates for any number of dimensions
            - Analytic surface partial derivatives of any order and number of dimensions
            - Analytic mean and gaussian curvatures
            - Isoparametric curves in the u- and v-directions

        The class can be used to represent polynomial and rational Bézier, B-Spline and NURBS surfaces
        The type of surface depends on the initialization arguments

            - Polymnomial Bézier: Provide the array of control points
            - Rational Bézier:    Provide the arrays of control points and weights
            - B-Spline:           Provide the array of control points, (u,v) degrees and (u,v) knot vectors
            - NURBS:              Provide the arrays of control points and weights, (u,v) degrees and (u,v) knot vectors

        In addition, this class supports operations with real and complex numbers
        The data type used for the computations is detected from the data type of the arguments
        Using complex numbers can be useful to compute the derivative of the shape using the complex step method


        References
        ----------
        The NURBS Book. See references to equations and algorithms throughout the code
        L. Piegl and W. Tiller
        Springer, second edition

        Curves and Surfaces for CADGD. See references to equations the source code
        G. Farin
        Morgan Kaufmann Publishers, fifth edition

        All references correspond to The NURBS book unless it is explicitly stated that they come from Farin's book


    """

    def __init__(self, control_points=None, weights=None, u_degree=None, v_degree=None, u_knots=None, v_knots=None):


        # Void initialization
        if control_points is None and weights is None and u_degree is None and v_degree is None \
                and u_knots is None and v_knots is None:
            pass


        # Polynomial Bezier surface initialization
        elif weights is None and u_degree is None and u_knots is None and v_degree is None and v_knots is None:

            # Set the surface type flag
            self.surface_type = 'Bezier'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1
            m = np.shape(control_points)[2] - 1

            # Define the weight of the control points
            weights = np.ones((n + 1, m + 1), dtype=control_points.dtype)

            # Define the order of the basis polynomials
            u_degree = n
            v_degree = m

            # Define the knot vectors (clamped spline)
            u_knots = np.concatenate((np.zeros(u_degree), np.linspace(0, 1, n - u_degree + 2), np.ones(u_degree)))
            v_knots = np.concatenate((np.zeros(v_degree), np.linspace(0, 1, m - v_degree + 2), np.ones(v_degree)))


        # Rational Bezier surface initialization
        elif u_degree is None and u_knots is None and v_degree is None and v_knots is None:

            # Set the surface type flag
            self.surface_type = 'R-Bezier'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1
            m = np.shape(control_points)[2] - 1

            # Define the order of the basis polynomials
            u_degree = n
            v_degree = m

            # Define the knot vectors (clamped spline)
            u_knots = np.concatenate((np.zeros(u_degree), np.linspace(0, 1, n - u_degree + 2), np.ones(u_degree)))
            v_knots = np.concatenate((np.zeros(v_degree), np.linspace(0, 1, m - v_degree + 2), np.ones(v_degree)))


        # B-Spline surface initialization
        elif weights is None:

            # Set the surface type flag
            self.surface_type = 'B-Spline'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1
            m = np.shape(control_points)[2] - 1

            # Define the weight of the control points
            weights = np.ones((n + 1, m + 1), dtype=control_points.dtype)


        # NURBS surface initialization
        else:

            # Set the surface type flag
            self.surface_type = 'NURBS'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]


        # Declare input variables as instance variables
        self.P = control_points
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots



    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute NURBS surface coordinates
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_value(self, u, v):

        """ Evaluate the coordinates of the surface corresponding to the (u,v) parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the coordinates of the surface
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check that u and v have the same size
        if np.isscalar(u) and np.isscalar(v): pass
        elif u.size == v.size: pass
        else: raise Exception('u and v must have the same size')

        # Evaluate the NURBS surface for the input (u,v) parametrization
        S = self.compute_nurbs_coordinates(self.P, self.W, self.p, self.q, self.U, self.V, u, v)

        return S


    @staticmethod
    def compute_nurbs_coordinates(P, W, p, q, U, V, u, v):

        """ Evaluate the coordinates of the NURBS surface corresponding to the (u,v) parametrization

        This function computes the coordinates of the NURBS surface in homogeneous space using equation 4.15 and then
        maps the coordinates to ordinary space using the perspective map given by equation 1.16. See algorithm A4.3

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        W : ndarray with shape (n+1, m+1)
            Array containing the weight of the control points
            The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
            The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the NURBS surface coordinates
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 3:           raise Exception('P must be an array of shape (ndim, n+1, m+1)')
        if W.ndim > 2:           raise Exception('W must be an array of shape (n+1, m+1)')
        if not np.isscalar(p):   raise Exception('p must be an scalar')
        if not np.isscalar(q):   raise Exception('q must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if V.ndim > 1:           raise Exception('V must be an array of shape (s+1=m+q+2,)')
        if np.isscalar(u):       u = np.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')
        if np.isscalar(v):       v = np.asarray(v)
        elif u.ndim > 1:         raise Exception('v must be a scalar or an array of shape (N,)')

        # Shape of the array of control points
        n_dim, nn, mm = np.shape(P)

        # Highest index of the control points (counting from zero)
        n = nn - 1
        m = mm - 1

        # Compute the B-Spline basis polynomials
        N_basis_u = compute_basis_polynomials(n, p, U, u)  # shape (n+1, N)
        N_basis_v = compute_basis_polynomials(m, q, V, v)  # shape (m+1, N)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((P * W[np.newaxis, :], W[np.newaxis, :]), axis=0)

        # Compute the coordinates of the NURBS surface in homogeneous space
        # This implementation is vectorized to increase speed
        A = np.dot(P_w, N_basis_v)                                      # shape (ndim+1, n+1, N)
        B = np.repeat(N_basis_u[np.newaxis], repeats=n_dim+1, axis=0)   # shape (ndim+1, n+1, N)
        S_w = np.sum(A*B, axis=1)                                       # shape (ndim+1, N)

        # Map the coordinates back to the ordinary space
        S = S_w[0:-1,:]/S_w[-1, :]

        return S


    @staticmethod
    def compute_bspline_coordinates(P, p, q, U, V, u, v):

        """ Evaluate the coordinates of the B-Spline surface corresponding to the (u,v) parametrization

        This function computes the coordinates of a B-Spline surface as given by equation 3.11. See algorithm A3.5

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the NURBS surface coordinates
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 3:           raise Exception('P must be an array of shape (ndim, n+1, m+1)')
        if not np.isscalar(p):   raise Exception('p must be an scalar')
        if not np.isscalar(q):   raise Exception('q must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if V.ndim > 1:           raise Exception('V must be an array of shape (s+1=m+q+2,)')
        if np.isscalar(u):       u = np.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')
        if np.isscalar(v):       v = np.asarray(v)
        elif u.ndim > 1:         raise Exception('v must be a scalar or an array of shape (N,)')

        # Shape of the array of control points
        n_dim, nn, mm = np.shape(P)

        # Highest index of the control points (counting from zero)
        n = nn - 1
        m = mm - 1

        # Compute the B-Spline basis polynomials
        N_basis_u = compute_basis_polynomials(n, p, U, u)  # shape (n+1, N)
        N_basis_v = compute_basis_polynomials(m, q, V, v)  # shape (m+1, N)

        # Compute the coordinates of the B-Spline surface
        # This implementation is vectorized to increase speed
        A = np.dot(P, N_basis_v)                                        # shape (ndim, n+1, N)
        B = np.repeat(N_basis_u[np.newaxis], repeats=n_dim, axis=0)     # shape (ndim, n+1, N)
        S = np.sum(A*B,axis=1)                                          # shape (ndim, N)

        return S


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the derivatives of the surface
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_derivative(self, u, v, order_u, order_v):

        """ Evaluate the derivative of the surface for the input u-parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        order_u : int
            Order of the partial derivative in the u-direction

        order_v : int
            Order of the partial derivative in the v-direction

        Returns
        -------
        dS : ndarray with shape (ndim, N)
            Array containing the derivative of the desired order
            The first dimension of ´dC´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´dC´ spans the ´u´ parametrization sample points

        """

        # Compute the array of surface derivatives up to the input (u,v) orders and slice the desired values
        dS = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v)[order_u, order_v, ...]

        return dS


    def compute_nurbs_derivatives(self, P, W, p, q, U, V, u, v, up_to_order_u, up_to_order_v):

        """ Compute the derivatives of a NURBS surface in ordinary space up to to the desired orders

        This function computes the analytic derivatives of the NURBS surface in ordinary space using equation 4.20 and
        the derivatives of the NURBS surface in homogeneous space obtained from compute_bspline_derivatives()

        The derivatives are computed recursively in a fashion similar to algorithm A4.4

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        W : ndarray with shape (n+1, m+1)
            Array containing the weight of the control points
            The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
            The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        up_to_order_u : integer
            Order of the highest derivative in the u-direction

        up_to_order_v : integer
            Order of the highest derivative in the v-direction

        Returns
        -------
        nurbs_derivatives: ndarray of shape (up_to_order_u+1, up_to_order_v+1, ndim, Nu)
            The first dimension spans the order of the u-derivatives (0, 1, 2, ...)
            The second dimension spans the order of the v-derivatives (0, 1, 2, ...)
            The third dimension spans the coordinates (x,y,z,...)
            The fourth dimension spans (u,v) parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u, v = np.asarray(u), np.asarray(v)
        if (P.dtype == np.complex128) or (W.dtype == np.complex128) or (u.dtype == np.complex128) or (v.dtype == np.complex128):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((P * W[np.newaxis, :], W[np.newaxis, :]), axis=0)

        # Compute the derivatives of the NURBS surface in homogeneous space
        bspline_derivatives = self.compute_bspline_derivatives(P_w, p, q, U, V, u, v, up_to_order_u, up_to_order_v)
        A_ders = bspline_derivatives[:, :, 0:-1, :]
        w_ders = bspline_derivatives[:, :, [-1], :]

        # Initialize array of derivatives
        n_dim, N = np.shape(P)[0], np.size(u)
        nurbs_derivatives = np.zeros((up_to_order_u+1, up_to_order_v+1, n_dim, N), dtype=data_type)

        # Compute the derivatives of up to the desired order
        # See algorithm A4.4 from the NURBS book
        for k in range(up_to_order_u+1):
            for L in range(up_to_order_v+1):

                # Update the numerator of equation 4.20 recursively
                temp_numerator = A_ders[[k], [L], ...]

                # Summation j=0 and point_index=1:k
                for i in range(1, k + 1):
                    temp_numerator -= scipy.special.binom(k, i)*w_ders[[i], [0], ...]*nurbs_derivatives[[k-i], [L], ...]

                # Summation point_index=0 and j=1:L
                for j in range(1, L + 1):
                    temp_numerator -= scipy.special.binom(L, j)*w_ders[[0], [j], ...]*nurbs_derivatives[[k], [L-j], ...]

                # Summation point_index=1:k and j=1:L
                for i in range(1, k+1):
                    for j in range(1, L+1):
                        temp_numerator -= scipy.special.binom(k, i) * scipy.special.binom(L, j)* w_ders[[i], [j], ...] * nurbs_derivatives[[k-i], [L-j], ...]

                # Compute the (k,L)-th order NURBS surface partial derivative
                nurbs_derivatives[k, L, ...] = temp_numerator/w_ders[[0], [0], ...]

        return nurbs_derivatives


    @staticmethod
    def compute_bspline_derivatives(P, p, q, U, V, u, v, up_to_order_u, up_to_order_v):

        """ Compute the derivatives of a B-Spline (or NURBS surface in homogeneous space) up to orders
        `derivative_order_u` and `derivative_order_v`

        This function computes the analytic derivatives of a B-Spline surface using equation 3.17. See algorithm A3.6

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        up_to_order_u : integer
            Order of the highest derivative in the u-direction

        up_to_order_v : integer
            Order of the highest derivative in the v-direction

        Returns
        -------
        bspline_derivatives: ndarray of shape (up_to_order_u+1, up_to_order_v+1, ndim, Nu)
            The first dimension spans the order of the u-derivatives (0, 1, 2, ...)
            The second dimension spans the order of the v-derivatives (0, 1, 2, ...)
            The third dimension spans the coordinates (x,y,z,...)
            The fourth dimension spans (u,v) parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u = np.asarray(u)
        if (P.dtype == np.complex128) or (u.dtype == np.complex128) or (v.dtype == np.complex128):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Set the B-Spline coordinates as the zero-th derivatives
        n_dim, N = np.shape(P)[0], np.size(u)
        bspline_derivatives = np.zeros((up_to_order_u+1, up_to_order_v+1, n_dim, N), dtype=data_type)

        # Compute the derivatives of up to the desired order
        # See algorithm A3.2 from the NURBS book
        for order_u in range(min(p, up_to_order_u) + 1):
            for order_v in range(min(q, up_to_order_v) + 1):

                # Highest index of the control points (counting from zero)
                n = np.shape(P)[1] - 1
                m = np.shape(P)[2] - 1

                # Compute the B-Spline basis polynomials
                N_basis_u = compute_basis_polynomials_derivatives(n, p, U, u, order_u)
                N_basis_v = compute_basis_polynomials_derivatives(m, q, V, v, order_v)

                # Compute the coordinates of the B-Spline surface
                # This implementation is vectorized to increase speed
                A = np.dot(P, N_basis_v)                                                # shape (ndim, n+1, N)
                B = np.repeat(N_basis_u[np.newaxis], repeats=n_dim, axis=0)             # shape (ndim, n+1, N)
                bspline_derivatives[order_u, order_v, :, :] = np.sum(A * B, axis=1)     # shape (ndim, N)

        # Note that derivatives with order higher than `p` and `q` are not computed and are be zero from initialization
        # These zero-derivatives are required to compute the higher order derivatives of rational surfaces

        return bspline_derivatives


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the isoparametric NURBS curves
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_isocurve_u(self, u0):

        """ Create a NURBS curve object that contains the surface isoparametric curve S(u0,v)

        The isoparametric nurbs curve is defined by equations 4.16 and 4.18

        Parameters
        ----------
        u0 : scalar
            Scalar defining the u-parameter of the isoparametric curve

        Returns
        -------
        isocurve_u : instance of NurbsCurve class
            Object defining the isoparametric curve

        """

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((self.P * self.W[np.newaxis, :], self.W[np.newaxis, :]), axis=0)

        # Compute the array of control points in homogeneous space
        n_dim, nn, mm = np.shape(P_w)
        n = nn - 1
        N_basis_u = compute_basis_polynomials(n, self.p, self.U, float(u0)).flatten()   # use float() to avoid problem with numba and integers
        N_basis_u = N_basis_u[np.newaxis, :, np.newaxis]
        Q_w = np.sum(P_w * N_basis_u, axis=1)

        # Compute the array of control points in ordinary space and the array of control point weights (inverse map)
        Q = Q_w[0:-1, :]/Q_w[-1, :]
        W = Q_w[-1, :]

        # Create the NURBS isoparametric curve in the u direction
        isocurve_u = NurbsCurve(control_points=Q, weights=W, degree=self.q, knots=self.V)

        return isocurve_u


    def get_isocurve_v(self, v0):

        """ Create a NURBS curve object that contains the surface isoparametric curve S(u,v0)

        The isoparametric nurbs curve is defined by equations 4.17 and 4.18

        Parameters
        ----------
        v0 : scalar
            Scalar defining the v-parameter of the isoparametric curve

        Returns
        -------
        isocurve_v : instance of NurbsCurve class
            Object defining the isoparametric curve

        """

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((self.P * self.W[np.newaxis, :], self.W[np.newaxis, :]), axis=0)

        # Compute the array of control points
        n_dim, nn, mm = np.shape(P_w)
        m = mm - 1
        N_basis_v = compute_basis_polynomials(m, self.q, self.V, float(v0)).flatten()   # use float() to avoid problem with numba and integers
        N_basis_v = N_basis_v[np.newaxis, np.newaxis, :]
        Q_w = np.sum(P_w * N_basis_v, axis=2)

        # Compute the array of control points in ordinary space and the array of control point weights (inverse map)
        Q = Q_w[0:-1, :]/Q_w[-1, :]
        W = Q_w[-1, :]

        # Create the NURBS isoparametric curve in the v direction
        isocurve_v = NurbsCurve(control_points=Q, weights=W, degree=self.p, knots=self.U)

        return isocurve_v


    # ---------------------------------------------------------------------------------------------------------------- #
    # Miscellaneous methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def attach_nurbs_udir(self, new_nurbs):

        """ Attatch a new NURBS curve to the end of the instance NURBS curve and return the merged NURBS curve"""

        # Check that the NURBS curves have the same degree
        if self.p != new_nurbs.p:
            raise Exception("In order to merge, the two NURBS curves must have the same degree")

        if self.q != new_nurbs.q:
            raise Exception("In order to merge, the two NURBS curves must have the same degree")

        if np.shape(self.V) != np.shape(new_nurbs.V):
            raise Exception("The two NURBS patches must have the same number of V-knots")

        if any([np.abs(v1-v2)>1e-12 for v1, v2 in zip(self.V, new_nurbs.V)]):
            raise Exception("The two NURBS patches must have the same V-knot values")

        # Combine the control points
        P = np.concatenate((self.P, new_nurbs.P), axis=1)

        # Combine the control point weights
        W = np.concatenate((self.W, new_nurbs.W), axis=0)

        # Highest index of the control points
        n1 = np.shape(self.P)[1] - 1
        n2 = np.shape(new_nurbs.P)[1] - 1

        # Combine the knot vectors (inner knot has p+1 multiplicity)
        U_start = np.zeros((self.p + 1,))
        U_end = np.ones((self.p + 1,))
        U_mid = np.ones((self.p + 1,)) / 2
        U1 = 0.00 + self.U[self.p + 1:n1 + 1] / 2
        U2 = 0.50 + new_nurbs.U[self.p + 1:n2 + 1] / 2
        U = np.concatenate((U_start, U1, U_mid, U2, U_end))

        # Create the merged NURBS surface
        mergedNurbs = NurbsSurface(control_points=P, weights=W, u_degree=self.p, v_degree=self.q, u_knots=U, v_knots=self.V)

        return mergedNurbs


    def attach_nurbs_vdir(self, new_nurbs):

        """ Attatch a new NURBS curve to the end of the instance NURBS curve and return the merged NURBS curve"""

        # Check that the NURBS curves have the same degree
        if self.p != new_nurbs.p:
            raise Exception("In order to merge, the two NURBS curves must have the same degree")

        if self.q != new_nurbs.q:
            raise Exception("In order to merge, the two NURBS curves must have the same degree")

        if np.shape(self.U) != np.shape(new_nurbs.U):
            raise Exception("The two NURBS patches must have the same number of V-knots")

        if any([np.abs(u1-u2)>1e-12 for u1, u2 in zip(self.U, new_nurbs.U)]):
            raise Exception("The two NURBS patches must have the same V-knot values")

        # Combine the control points
        P = np.concatenate((self.P, new_nurbs.P), axis=2)

        # Combine the control point weights
        W = np.concatenate((self.W, new_nurbs.W), axis=1)

        # Highest index of the control points
        n1 = np.shape(self.P)[2] - 1
        n2 = np.shape(new_nurbs.P)[2] - 1

        # Combine the knot vectors (inner knot has p+1 multiplicity)
        V_start = np.zeros((self.q + 1,))
        V_end = np.ones((self.q + 1,))
        V_mid = np.ones((self.q + 1,)) / 2
        V1 = 0.00 + self.U[self.q + 1:n1 + 1] / 2
        V2 = 0.50 + new_nurbs.U[self.q + 1:n2 + 1] / 2
        V = np.concatenate((V_start, V1, V_mid, V2, V_end))

        # Create the merged NURBS surface
        mergedNurbs = NurbsSurface(control_points=P, weights=W, u_degree=self.p, v_degree=self.q, u_knots=self.U, v_knots=V)

        return mergedNurbs


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the unitary normal vectors
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_normals(self, u, v):

        """ Evaluate the unitary vectors normal to the surface the input (u,v) parametrization

        The definition of the unitary normal vector is given in section 19.2 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the normals

        v : scalar or ndarray with shape (N,)
            Scalar or array containing the v-parameter used to evaluate the normals

        Returns
        -------
        normals : ndarray with shape (ndim, N)
            Array containing the unitary vectors normal to the surface

        """

        # Compute 2 vectors tangent to the surface
        S_u = self.get_derivative(u, v, order_u=1, order_v=0)
        S_v = self.get_derivative(u, v, order_u=0, order_v=1)

        # Compute the normal vector as the cross product of the tangent vectors and normalize it
        normals = np.cross(S_u, S_v, axisa=0, axisb=0, axisc=0)
        normals = normals/np.sum(normals ** 2, axis=0) ** (1 / 2)

        return normals



    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the mean and Gaussian curvatures
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_curvature(self, u, v):

        """ Evaluate the mean and gaussian curvatures of the surface the input (u,v) parametrization

        The definition of the gaussian and mean curvatures are given by equations 19.11 and 19.12 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the curvatures

        v : scalar or ndarray with shape (N,)
            Scalar or array containing the v-parameter used to evaluate the curvatures

        Returns
        -------
        mean_curvature : ndarray with shape (N, )
            Scalar or array containing the mean curvature of the surface

        gaussian_curvature : ndarray with shape (N, )
            Scalar or array containing the gaussian curvature of the surface

        """

        # Compute the partial derivatives
        S_u = self.get_derivative(u, v, order_u=1, order_v=0)
        S_v = self.get_derivative(u, v, order_u=0, order_v=1)
        S_uu = self.get_derivative(u, v, order_u=2, order_v=0)
        S_uv = self.get_derivative(u, v, order_u=1, order_v=1)
        S_vv = self.get_derivative(u, v, order_u=0, order_v=2)

        # Compute the normal vector
        N = self.get_normals(u, v)

        # Compute the components of the first fundamental form of the surface
        E = np.sum(S_u * S_u, axis=0)
        F = np.sum(S_u * S_v, axis=0)
        G = np.sum(S_v * S_v, axis=0)

        # Compute the components of the second fundamental form of the surface
        L = np.sum(S_uu * N, axis=0)
        M = np.sum(S_uv * N, axis=0)
        N = np.sum(S_vv * N, axis=0)

        # Compute the mean curvature
        mean_curvature = (1/2) * (N * E - 2 * M * F + L * G) / (E * G - F ** 2)

        # Compute the gaussian curvature
        gaussian_curvature = (L * N - M ** 2) / (E * G - F ** 2)

        return mean_curvature, gaussian_curvature



    # ---------------------------------------------------------------------------------------------------------------- #
    # Plotting functions
    # ---------------------------------------------------------------------------------------------------------------- #
    def plot(self, fig=None, ax = None,
             surface=True, surface_color='blue', colorbar=False,
             boundary=True, control_points=False, normals=False, axis_off=False, ticks_off=False,
             Nu=50, Nv=50, isocurves_u=None, isocurves_v=None):

        # Prepare the plot
        if fig is None:
            fig = mpl.pyplot.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(azim=-105, elev=30)
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('k')
            ax.yaxis.pane.set_edgecolor('k')
            ax.zaxis.pane.set_edgecolor('k')
            ax.xaxis.pane._alpha = 0.9
            ax.yaxis.pane._alpha = 0.9
            ax.zaxis.pane._alpha = 0.9
            ax.set_xlabel('$x$ axis', fontsize=11, color='k', labelpad=18)
            ax.set_ylabel('$y$ axis', fontsize=11, color='k', labelpad=18)
            ax.set_zlabel('$z$ axis', fontsize=11, color='k', labelpad=18)
            # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            # ax_xy.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(8)
            for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(8)
            for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(8)
            ax.xaxis.set_rotate_label(False)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            if ticks_off:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
            if axis_off:
                ax.axis('off')

        # Add objects to the plot
        if surface:        self.plot_surface(fig, ax, color=surface_color, colorbar=colorbar, Nu=Nu, Nv=Nv)
        if boundary:       self.plot_boundary(fig, ax)
        if control_points: self.plot_control_points(fig, ax)
        if normals:        self.plot_normals(fig, ax)
        if isocurves_u:
            self.plot_isocurve_u(fig, ax, u_values=np.linspace(0, 1, isocurves_u))
        if isocurves_v:
              self.plot_isocurve_v(fig, ax, v_values=np.linspace(0, 1, isocurves_v))

        # Set the scaling of the axes
        self.rescale_plot(fig, ax)

        return fig, ax


    def plot_surface(self, fig, ax, color='blue', alpha=0.30, colorbar=False, Nu=50, Nv=50):

        # Get the surface coordinates
        u = np.linspace(0.00, 1.00, Nu)
        v = np.linspace(0.00, 1.00, Nv)
        [uu, vv] = np.meshgrid(u, v, indexing='ij')
        u = uu.flatten()
        v = vv.flatten()
        X, Y, Z = np.real(self.get_value(u, v)).reshape((3, Nu, Nv))

        # Plot the surface
        if color == 'mean_curvature':

            # Define a colormap based on the curvature values
            mean_curvature, _ = np.real(self.get_curvature(u, v))
            curvature = np.reshape(mean_curvature, (Nu, Nv))
            curvature_normalized = (curvature - np.amin(curvature)) / (np.amax(curvature) - np.amin(curvature))
            curvature_colormap = mpl.cm.viridis(curvature_normalized)

            # Plot the surface with a curvature colormap
            surf_handle = ax.plot_surface(X, Y, Z,
                                          # color='blue',
                                          # edgecolor='blue',
                                          # cmap = 'viridis',
                                          facecolors=curvature_colormap,
                                          linewidth=0.75,
                                          alpha=1,
                                          shade=False,
                                          antialiased=True,
                                          zorder=2,
                                          ccount=Nu,
                                          rcount=Nv)
            if colorbar:
                fig.set_size_inches(7, 5)
                surf_handle.set_clim(np.amin(curvature), np.amax(curvature))
                cbar = fig.colorbar(surf_handle, ax=ax, orientation='vertical', pad=0.15, fraction=0.03, aspect=20)
                cbar.set_label(color)

        elif color == 'gaussian_curvature':

            # Define a colormap based on the curvature values
            _, gaussian_curvature= np.real(self.get_curvature(u, v))
            curvature = np.reshape(gaussian_curvature, (Nu, Nv))
            curvature_normalized = (curvature - np.amin(curvature)) / (np.amax(curvature) - np.amin(curvature))
            curvature_colormap = mpl.cm.viridis(curvature_normalized)

            # Plot the surface with a curvature colormap
            surf_handle = ax.plot_surface(X, Y, Z,
                                          # color='blue',
                                          # edgecolor='blue',
                                          # cmap = 'viridis',
                                          facecolors=curvature_colormap,
                                          linewidth=0.75,
                                          alpha=1,
                                          shade=False,
                                          antialiased=True,
                                          zorder=2,
                                          ccount=Nu,
                                          rcount=Nv)
            if colorbar:
                fig.set_size_inches(7, 5)
                surf_handle.set_clim(np.amin(curvature), np.amax(curvature))
                cbar = fig.colorbar(surf_handle, ax=ax, orientation='vertical', pad=0.15, fraction=0.03, aspect=20)
                cbar.set_label(color)

        else:

            # Plot the surface with a plain color
            ax.plot_surface(X, Y, Z,
                            color=color,
                            # edgecolor='blue',
                            linewidth=0,
                            alpha=alpha,
                            shade=False,
                            antialiased=True,
                            zorder=0,
                            ccount=Nv,
                            rcount=Nu)


    def plot_boundary(self, fig, ax, color='black', linewidth=1.00, linestyle='-',
                      south=True, north=True, east=True, west=True):

        """ Plot the isoparametric curves at the boundary """

        # Create the isoparametric NURBS curves and plot them on the current figure
        if east: self.get_isocurve_u(u0=0.00).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
        if west: self.get_isocurve_u(u0=1.00).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
        if south: self.get_isocurve_v(v0=0.00).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)
        if north: self.get_isocurve_v(v0=1.00).plot_curve(fig, ax, linestyle=linestyle, linewidth=linewidth, color=color)


    def plot_isocurve_u(self, fig, ax, u_values, color='black', linewidth=1.00, linestyle='-'):

        """ Plot isoparametric curves in the u-direction """
        for u in u_values: self.get_isocurve_u(u0=u).plot_curve(fig, ax, color=color, linewidth=linewidth, linestyle=linestyle)


    def plot_isocurve_v(self, fig, ax, v_values, color='black', linewidth=1.00, linestyle='-'):

        """ Plot isoparametric curves in the v-direction """
        for v in v_values: self.get_isocurve_v(v0=v).plot_curve(fig, ax, color=color, linewidth=linewidth, linestyle=linestyle)


    def plot_control_points(self, fig, ax, color='red', linewidth=1.00, linestyle='-', markersize=5, markerstyle='o'):

        """ Plot the control points """

        # Plot the control net
        Px, Py, Pz = np.real(self.P)
        ax.plot_wireframe(Px, Py, Pz,
                          edgecolor=color,
                          linewidth=linewidth,
                          linestyles=linestyle,
                          alpha=1.0,
                          antialiased=True,
                          zorder=1)

        # Plot the control points
        points, = ax.plot(Px.flatten(), Py.flatten(), Pz.flatten())
        points.set_linewidth(linewidth)
        points.set_linestyle(' ')
        points.set_marker(markerstyle)
        points.set_markersize(markersize)
        points.set_markeredgewidth(linewidth)
        points.set_markeredgecolor(color)
        points.set_markerfacecolor('w')
        points.set_zorder(4)
        # points.set_label(' ')


    def plot_normals(self, fig, ax, number_u=10, number_v=10, scale=0.075):

        """ Plot the normal vectors """

        # Compute the surface coordinates and normal vectors
        h = 1e-6 # Add a small offset to avoid poles at the extremes [0, 1]
        u = np.linspace(0.00+h, 1.00-h, number_u)
        v = np.linspace(0.00+h, 1.00-h, number_v)
        [u, v] = np.meshgrid(u, v, indexing='xy')
        u = u.flatten()
        v = v.flatten()
        S = np.real(self.get_value(u, v))
        N = np.real(self.get_normals(u, v))

        # Scale the normal vectors and plot them
        Lu = self.get_isocurve_u(u0=0.50).get_arclength()
        Lv = self.get_isocurve_v(v0=0.50).get_arclength()
        length_scale = scale*np.real(np.amax([Lu, Lv]))
        N = length_scale * N
        ax.quiver(S[0, :], S[1, :], S[2, :], N[0, :], N[1, :], N[2, :], color='black', length=np.abs(scale), normalize=True)


    def rescale_plot(self, fig, ax):

        """ Adjust the aspect ratio of the figure """

        # Set axes aspect ratio
        ax.autoscale(enable=True)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        z_min, z_max = ax.get_zlim()
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        L = np.max((x_max - x_min, y_max - y_min, z_max - z_min)) / 2
        ax.set_xlim3d(x_mid - 1.0 * L, x_mid + 1.0 * L)
        ax.set_ylim3d(y_mid - 1.0 * L, y_mid + 1.0 * L)
        ax.set_zlim3d(z_mid - 1.0 * L, z_mid + 1.0 * L)

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)


    def plot_curvature(self, fig=None, ax=None, curvature_type='mean'):

        # Prepare the plot
        if fig is None:
            fig = mpl.pyplot.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=-105, elev=30)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
        ax.xaxis.pane._alpha = 0.9
        ax.yaxis.pane._alpha = 0.9
        ax.zaxis.pane._alpha = 0.9
        ax.set_xlabel('$x$ axis', fontsize=11, color='k', labelpad=18)
        ax.set_ylabel('$y$ axis', fontsize=11, color='k', labelpad=18)
        ax.set_zlabel('$z$ axis', fontsize=11, color='k', labelpad=18)
        # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(8)
        for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(8)
        for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(8)
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        # ax_xy.set_xticks([])
        # ax_xy.set_yticks([])
        # ax_xy.set_zticks([])
        # ax_xy.axis('off')

        # (u,v) parametrization for the plot
        Nu, Nv = 50, 50
        u = np.linspace(0.00, 1.00, Nu)
        v = np.linspace(0.00, 1.00, Nv)
        [uu, vv] = np.meshgrid(u, v, indexing='ij')
        u = uu.flatten()
        v = vv.flatten()

        # Get the curvature
        if curvature_type == 'mean':
            curvature, _ = np.real(self.get_curvature(u, v))
        elif curvature_type == 'gaussian':
            _, curvature = np.real(self.get_curvature(u, v))
        else:
            raise Exception("Choose a valid curvature type: 'mean' or 'gaussian'")

        # Represent the curvature as a carpet plot or as a surface plot
        ax.set_xlabel('$u$', fontsize=11, color='k', labelpad=10)
        ax.set_ylabel('$v$', fontsize=11, color='k', labelpad=10)
        ax.set_zlabel(r'$\kappa$' + ' ' + curvature_type, fontsize=11, color='k', labelpad=20)
        curvature = np.reshape(curvature, (Nu, Nv))
        ax.plot_surface(uu, vv, curvature,
                        color='blue',
                        # edgecolor='blue',
                        # cmap = 'viridis',
                        # facecolors=curvature_colormap,
                        linewidth=0.75,
                        alpha=0.50,
                        shade=False,
                        antialiased=True,
                        zorder=0,
                        ccount=50,
                        rcount=50)

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

        return fig, ax
