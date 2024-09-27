import numpy as np

def pendulum_time_invariant(xy, t, l=1, g=9.81, a=2.0):
    """
    Calculate the dynamics of a pendulum in phase space.

    This function describes the motion of a pendulum using a system of ordinary differential equations (ODEs). 
    The pendulum's dynamics can be expressed as:
    theta_dotdot = -(a/l) * theta_dot - (g/l) * sin(theta)

    For small angles, the equation is linearized as:
    theta_dotdot = -(a/l) * theta_dot - (g/l) * theta

    In the phase space, the system can be represented in 2D where:
    x = angular velocity
    y = angular position

    This leads to the representation:
    xy_dot = A * xy

    Parameters:
    - xy (tuple): A tuple containing values (x, y), where x is the angular velocity and y is the angular position.
    - l (float): Length of the pendulum. Default is 1.
    - g (float): Gravitational acceleration. Default is 9.81.
    - a (float): Damping coefficient. Default is 1.

    Returns:
    - numpy.array: Returns an array [x_dot, y_dot] representing the rate of change of angular velocity and position.
    """
    x,y = xy
    x_dot = -a/l * x - g/l * y
    y_dot = x
    return np.array([x_dot, y_dot])

def lorenz(xyz, t, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])