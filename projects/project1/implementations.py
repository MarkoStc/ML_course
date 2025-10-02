import numpy as np

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def compute_mse(y, tx, w):
    """Compute mean squared error."""
    e = y - tx @ w
    return (e @ e) / (2 * len(y))


def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def compute_logistic_loss(y, tx, w):
    """Logistic regression negative log likelihood."""
    pred = sigmoid(tx @ w)
    # Avoid log(0)
    eps = 1e-15
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))


def compute_logistic_gradient(y, tx, w):
    """Gradient for logistic regression."""
    pred = sigmoid(tx @ w)
    return tx.T @ (pred - y) / len(y)


# ------------------------------------------------------------
# Linear regression
# ------------------------------------------------------------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = -(tx.T @ (y - tx @ w)) / len(y)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1)."""
    w = initial_w
    n = len(y)
    for i in range(max_iters):
        idx = i % n
        xi = tx[idx:idx+1]
        yi = y[idx:idx+1]
        grad = -(xi.T @ (yi - xi @ w)) / 1
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.inv(tx.T @ tx) @ (tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    A = tx.T @ tx + 2 * n * lambda_ * np.identity(d)
    b = tx.T @ y
    w = np.linalg.inv(A) @ b
    loss = compute_mse(y, tx, w)
    return w, loss


# ------------------------------------------------------------
# Logistic regression
# ------------------------------------------------------------
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    w = initial_w
    n = len(y)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w) + lambda_ * (w @ w)
    return w, loss
