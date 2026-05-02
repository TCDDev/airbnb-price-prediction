"""
OLS Linear Regression (from scratch)

We model:

    y = β₀ + β₁x₁ + ... + β_d x_d

By adding a column of ones to X, we can write:

    y = Xβ

and solve for β using:

    β = (XᵀX)⁻¹ Xᵀy   (normal equation)

In practice, we use the pseudoinverse for stability (XᵀX may be singular or numerically unstable):

    β = pinv(X) y

This keeps the implementation fully vectorized and simple.
"""

import numpy as np

class OLSLinearRegression:
    def __init__(self):
        self.beta = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
    
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        self.beta = np.linalg.pinv(X_b) @ y
        return self
    
    def predict(self, X):
        if self.beta is None:
            raise ValueError("Model has not yet been fit.")
        
        X = np.asarray(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta
    
if __name__ == '__main__':
    # Testing the class with random data
    np.random.seed(42)

    n_samples = 100
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([3.0, 2.0, -1.0]) # [β₀, β₁, β₂]
    X_b = np.c_[np.ones((n_samples, 1)), X]

    y = X_b @ true_beta + np.random.randn(n_samples) * 0.1

    model = OLSLinearRegression()
    model.fit(X, y)

    print(f"True Beta:\t{true_beta}")
    print(f"Estimated Beta:\t{model.beta}")