import numpy as np

def multi_regress(y, Z):
    """
    Perform multiple linear regression using normal equations (lecture notation).

    Args:
        y (np.ndarray): Dependent variable vector (n_samples,)
        Z (np.ndarray): Design matrix (n_samples, n_features)

    Returns:
        a (np.ndarray): Coefficient vector (n_features,)
        residuals (np.ndarray): Residuals (y - Z a)
        r_squared (float): Coefficient of determination R²
    """
    A = Z.T @ Z
    Y = Z.T @ y
    a = np.linalg.solve(A, Y)

    y_pred = Z @ a
    residuals = y - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r_squared = 1 - (ss_res / ss_tot)

    return a, residuals, r_squared
