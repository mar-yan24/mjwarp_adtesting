"""Finite-difference gradient and Jacobian utilities."""

import numpy as np


def fd_gradient(fn, x_np, eps=1e-3):
  """Central-difference gradient of scalar fn w.r.t. flat array x_np.

  Args:
    fn: Callable that takes a numpy array and returns a scalar.
    x_np: Point at which to evaluate the gradient.
    eps: Perturbation size.

  Returns:
    Gradient array with same shape as x_np.
  """
  grad = np.zeros_like(x_np)
  for i in range(x_np.size):
    x_plus = x_np.copy()
    x_minus = x_np.copy()
    x_plus.flat[i] += eps
    x_minus.flat[i] -= eps
    grad.flat[i] = (fn(x_plus) - fn(x_minus)) / (2.0 * eps)
  return grad


def fd_jacobian(fn, x_np, n_outputs, eps=1e-3):
  """Central-difference Jacobian for vector-valued fn.

  Args:
    fn: Callable that takes numpy array, returns numpy array of shape (n_outputs,).
    x_np: Point at which to evaluate the Jacobian.
    n_outputs: Number of output dimensions.
    eps: Perturbation size.

  Returns:
    Jacobian matrix of shape (n_outputs, x_np.size).
  """
  jac = np.zeros((n_outputs, x_np.size))
  for i in range(x_np.size):
    x_plus = x_np.copy()
    x_minus = x_np.copy()
    x_plus.flat[i] += eps
    x_minus.flat[i] -= eps
    jac[:, i] = (fn(x_plus) - fn(x_minus)) / (2.0 * eps)
  return jac


def taylor_test(fn, grad_fn, x_np, direction=None, h_values=None):
  """Taylor convergence test for gradient correctness.

  For a scalar function f with gradient g, verifies that
  |f(x + h*d) - f(x) - h * g^T * d| = O(h^2)

  Args:
    fn: Scalar function f(x).
    grad_fn: Returns gradient g at x (same shape as x).
    x_np: Point to test at.
    direction: Perturbation direction (random unit vector if None).
    h_values: Sequence of step sizes to test.

  Returns:
    Tuple of (h_values, remainders) for convergence analysis.
  """
  if h_values is None:
    h_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

  if direction is None:
    rng = np.random.RandomState(42)
    direction = rng.randn(*x_np.shape)
    direction = direction / np.linalg.norm(direction)

  f0 = fn(x_np)
  g = grad_fn(x_np)
  directional_deriv = np.dot(g.ravel(), direction.ravel())

  remainders = []
  for h in h_values:
    f_perturbed = fn(x_np + h * direction)
    remainder = abs(f_perturbed - f0 - h * directional_deriv)
    remainders.append(remainder)

  return np.array(h_values), np.array(remainders)
