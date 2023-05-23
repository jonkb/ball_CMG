""" Functions related to differentiation & differential equations
Maybe in the future: more linearization & surrogate modeling
"""

import numpy as np

## Constants
eps = np.finfo(float).eps # Machine Epsilon
tiny = np.finfo(float).tiny
h_fd = eps * 10**6 # May be too small

def FD(f, x0, f0=None, h=h_fd, scalar=False):
  """ Finite-difference differentiation (forward difference)
  Returns jacobian of F evaluated at x0

  Parameters
  ----------
  scalar (bool): Whether the independant variable x is a scalar.

  Returns
  -------
  Jacobian of f
    if f maps R --> R, then this is a scalar (assuming scalar=True)
    if f maps R^n --> R^m, then this is a (m,n) np array (Jacobian matrix)
    if f maps R^n --> R, then this is a (m,) np array (Gradient vector)
        Note that this is not a column vector.
  """
  
  if f0 is None:
    f0 = f(x0)
  
  if scalar:
    f1 = f(x0 + h)
    return (f1 - f0) / h

  # These next two lines make a matrix where each row represents a small
  #   step in that direction by h
  steps = np.eye(x0.size) * h
  x1 = x0 + steps
  # f1 = f(x1)
  f1 = np.array([f(row) for row in x1])
  # Forward-difference
  grad = (f1 - f0) / h
  # The Jacobian is grad.T
  return grad.T

def rk4(f, x, dt, prms=()):
  """ 4th order Runge-Kutta integration step
  Returns x(t+dt)
  """
  F1 = f(x, *prms)
  F2 = f(x + dt / 2 * F1, *prms)
  F3 = f(x + dt / 2 * F2, *prms)
  F4 = f(x + dt * F3, *prms)
  return x + dt / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
