"""Math utility functions."""

def exact_div(a, b):
    """Return a / b as an integer, but only if a is exactly divisible by b."""
    q = a // b
    if q * b != a:
        raise ValueError(f"{a} is not exactly divisible by {b}")
    return q

def interpolate_nans(x):
    """Interpolate NaN values in a numpy array."""
    import numpy as np
    
    if isinstance(x, list):
        x = np.array(x)
    
    nans = np.isnan(x)
    if not np.any(nans):
        return x
    
    # Find valid indices
    valid = ~nans
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) == 0:
        # All values are NaN
        return x
    
    # Interpolate
    x[nans] = np.interp(
        np.where(nans)[0],
        valid_indices,
        x[valid_indices]
    )
    
    return x
