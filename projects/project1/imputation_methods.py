"""Some imputation methods for project 1."""


import numpy as np

def impute_median(data: np.ndarray) -> np.ndarray:
    """
    Imputes NaN values in a 2D numpy array with the median of their respective column.
    
    Parameters:
        data (np.ndarray): 2D numpy array with possible NaN values.
        
    Returns:
        np.ndarray: A new array with NaNs replaced by column medians.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if data.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")
    
    # Copy to avoid modifying original data
    result = data.copy().astype(float)
    
    # Compute medians ignoring NaNs
    col_medians = np.nanmedian(result, axis=0)
    
    # Find NaN positions
    inds = np.where(np.isnan(result))
    
    # Replace NaNs with corresponding column medians
    result[inds] = np.take(col_medians, inds[1])
    
    return result










