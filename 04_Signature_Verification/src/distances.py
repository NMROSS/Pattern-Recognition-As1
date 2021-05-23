import numpy as np
from dtaidistance import dtw_ndim


def calculate_distance(signature_test, signatures_real) -> np.array:
    """
    Calculate the distances between a signature and and array of signatures

    :return: Array of distance to each genuine signature  np.array
    """
    num_signatures = len(signatures_real)
    distances = np.zeros((num_signatures, 1))

    # join signature features e.g (x, y) or (x, y, vy, vx) returned a list of feature arrays
    signature_features = [
        np.stack((sig.features()), axis=1) for sig in signatures_real
    ]
    test_features = np.stack((signature_test.features()), axis=1)

    # Calculate test signature against true signatures
    for i, sig in enumerate(signature_features):
        distances[i] = dtw_ndim.distance_fast(test_features, sig)

    return distances


# TODO: Can we have a smal documentation what treshold exactly is here?
def predict_fake(unverified_signatures, real_signatures, threshold=4, verbose=False):
    # compare signature against all genuine
    distances = calculate_distance(unverified_signatures, real_signatures)

    distances_mean = np.mean(distances)

    # classify signature as genuine if less the threshold set
    prediction = distances_mean >= threshold

    # compare our prediction with ground truth
    correct_prediction = prediction == unverified_signatures.is_fake

    # print each comparison
    if verbose:
        print('Ground Truth = {0}, Predicted = {1}'.format(unverified_signatures.is_fake, prediction))

    return correct_prediction
