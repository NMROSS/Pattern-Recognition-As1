import numpy as np
from dtaidistance import dtw_ndim


def mean_distance(signatures):
    # Join features together e.g. (x, y) OR (x, y, pressure)
    signature_features = [np.stack((sig.normalise().x, sig.normalise().y), axis=1) for sig in signatures]
    signature_features = np.asarray(signature_features)

    # Calculate distance matrix NOTE: unsure if axis is 1 OR 0
    distance_matrix = dtw_ndim.distance_matrix(signature_features, 1)

    # Flatten and remove duplicate distances
    #  0, 1, 2
    #  1, 0, 3  = [1,2,3]
    #  2, 3, 0
    distances = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    return np.mean(distances)
