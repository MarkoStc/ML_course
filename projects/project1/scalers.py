import numpy as np

class NormalScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Similar to sklearn.preprocessing.StandardScaler.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def fit_scale(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler on the training data (compute mean and std) 
        and return the scaled data.
        
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Scaled array with zero mean and unit variance.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        
        self.fitted = True
        return (X - self.mean_) / self.std_

    def scale(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data using mean and std from training data.
        
        Parameters
        ----------
        X : np.ndarray
            2D numpy array of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Scaled array using stored mean and std.
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_scale first.")
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if X.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        
        return (X - self.mean_) / self.std_