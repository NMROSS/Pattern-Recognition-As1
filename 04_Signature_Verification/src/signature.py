import numpy as np
from sklearn.preprocessing import minmax_scale


def read_from_file(path: str, signature_id: int):
    """
    Reads a signature from a given path with given ID.

    :param path: The path of the file to parse
    :param signature_id: The ID of the signature
    :return: Signature
    """
    with open(path, 'r') as f:
        raw_lines = f.readlines()

        lines = np.zeros([len(raw_lines), 7], dtype=np.double)

        for i in range(len(raw_lines)):
            line = np.fromstring(raw_lines[i], dtype=np.double, sep=" ")
            assert line.shape[0] == 7
            lines[i] = line

        t = lines[:, 0]
        x = lines[:, 1]
        y = lines[:, 2]
        pressure = lines[:, 3]
        penup = lines[:, 4]
        azimuth = lines[:, 5]
        inclination = lines[:, 6]

        return Signature(signature_id, t, x, y, pressure, penup, azimuth, inclination)


class Signature:
    """
    A signature consists of multiple data in SoA (struct of arrays) format. Each array element links with the elements
    of the other arrays at the same index.

    * `t` - timestamps
    * `x` - x position at t
    * `y` - y position at t
    * `pressure` - pressure at t
    * `penup` - Was the pen up (not pressed down) at t
    * `azimuth` - TODO
    * `inclination` - TODO
    """

    def __init__(
            self,
            id: int,
            t: np.array,
            x: np.array,
            y: np.array,
            pressure: np.array,
            penup: np.array,
            azimuth: np.array,
            inclination: np.array,
            is_fake: bool = False,
    ):
        assert len(t) == len(x)
        assert len(t) == len(y)
        assert len(t) == len(pressure)
        assert len(t) == len(penup)
        assert len(t) == len(azimuth)
        assert len(t) == len(inclination)

        self.id = id
        self.t = t
        self.x = x
        self.y = y
        self.pressure = pressure
        self.penup = penup
        self.azimuth = azimuth
        self.inclination = inclination
        self.is_fake = is_fake

    def to_dtw(self):
        size = self.x.shape[0] - 1
        vx = np.empty(size)
        vy = np.empty(size)

        for curr in range(size):
            next = curr + 1

            dt = self.t[next] - self.t[curr]
            dx = self.x[next] - self.x[curr]
            vx[curr] = dx / dt

            dy = self.y[next] - self.y[curr]
            vy[curr] = dy / dt

        return SignatureDTW(
            self.id,
            minmax_scale(self.x, feature_range=(-1, 1)),
            minmax_scale(self.y, feature_range=(-1, 1)),
            minmax_scale(vx, feature_range=(-1, 1)),
            minmax_scale(vy, feature_range=(-1, 1)),
            minmax_scale(self.pressure, feature_range=(0, 1)),
            self.is_fake,
        )

    def features(self) -> (np.array, np.array):
        """
        Returns selected feature combinations e.g. (x, y) OR (vx, vy, pressure)
        Values are normalised
        """
        return self.to_dtw().x, self.to_dtw().y


class SignatureDTW:
    """
    A signature in dynamic time warp format consists of multiple normalized data in SoA (struct of arrays) format.

    * `x` - x positions (normalized between [-1, 1])
    * `y` - y positions (normalized between [-1, 1])
    * `vx` - x velocities (length one less than `x`) (normalized between [-1, 1])
    * `vy` - y velocities (length one less than `x`) (normalized between [-1, 1])
    * `pressure` - pressures (normalized between [0, 1])
    """

    def __init__(
            self,
            id: int,
            x: np.array,
            y: np.array,
            vx: np.array,
            vy: np.array,
            pressure: np.array,
            is_fake: bool = False,
    ):
        assert len(x) == len(y)
        assert len(x) == len(pressure)
        assert len(vx) == len(vy)

        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.pressure = pressure
        self.is_fake = is_fake
