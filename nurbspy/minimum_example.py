# NURBS curve minimum working example
def run():

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
