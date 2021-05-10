import numpy as np


def upper_contour(img: np.array) -> np.array:
    cols = np.full(img.shape[1], img.shape[0], dtype=int)

    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            if not img[row, col]:
                cols[col] = row
                break

    return cols


def lower_contour(img: np.array) -> np.array:
    cols = np.full(img.shape[1], img.shape[0], dtype=int)

    for col in range(img.shape[1]):
        for row in range(img.shape[0]).__reversed__():
            if not img[row, col]:
                cols[col] = row
                break

    return cols


def num_transitions(img: np.array) -> np.array:
    cols = np.zeros(img.shape[1], dtype=int)

    for col in range(img.shape[1]):
        empty = img[0, col]

        for row in range(img.shape[0]):
            if img[row, col] != empty:
                cols[col] += 1
                empty = not empty

    return cols


def fraction_black(img: np.array) -> np.array:
    cols = np.empty(img.shape[1], dtype=float)

    for col in range(img.shape[1]):
        blacks = 0

        for row in range(img.shape[0]):
            if not img[row, col]:
                blacks += 1

        cols[col] = blacks / img.shape[0]

    return cols


def fraction_black_in_contour(img: np.array) -> np.array:
    lower = lower_contour(img)
    upper = upper_contour(img)

    cols = np.zeros(img.shape[1], dtype=float)

    for col in range(img.shape[1]):
        if abs(upper[col] - lower[col]) == 0:
            continue

        blacks = 0

        for row in range(upper[col], lower[col]):
            if not img[row, col]:
                blacks += 1

        if blacks != 0:
            cols[col] = blacks / abs(upper[col] - lower[col])

    return cols
