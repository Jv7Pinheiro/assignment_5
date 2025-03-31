import numpy as np

def discretize_data(data, n=2, bool=True, scale=0.0):
    # If scale is not zero, then we want to scale the data by scale
    # If scale is less than zero, ignore it
    if (scale > 0.0):
        data *= scale
    

    # Sort data and determine thresholds
    sorted_indices = np.argsort(data)
    thresholds = np.linspace(0, len(data), n+1, dtype=float)
    
    # Create labels based on partition
    labels = np.zeros(len(data), dtype=float)
    for i in range(n):
        labels[sorted_indices[thresholds[i]:thresholds[i+1]]] = i
    
    return labels