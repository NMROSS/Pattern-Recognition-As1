import numpy as np
from sklearn.preprocessing import minmax_scale


def read_from_file(file: str):
    with open(file, 'r') as f:
        raw_lines = f.readlines()

        lines = np.zeros([len(raw_lines), 7], dtype=np.float32)

        for i in range(len(raw_lines)):
            line = np.fromstring(raw_lines[i], dtype=np.float32, sep=" ")
            assert line.shape[0] == 7
            lines[i] = line

        t = lines[:, 0]
        x = lines[:, 1]
        y = lines[:, 2]
        pressure = lines[:, 3]
        penup = lines[:, 4]
        azimuth = lines[:, 5]
        inclination = lines[:, 6]

        return Signature(t, x, y, pressure, penup, azimuth, inclination)


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

        self.t = t
        self.x = x
        self.y = minmax_scale(y, feature_range=(-1, 1))
        self.pressure = minmax_scale(pressure, feature_range=(0, 1))
        self.penup = penup
        self.azimuth = azimuth
        self.inclination = inclination
        self.is_fake = is_fake

    def dtw(self):
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

        SignatureDTW(
            minmax_scale(self.x, feature_range=(-1, 1)),
            minmax_scale(self.y, feature_range=(-1, 1)),
            minmax_scale(vx, feature_range=(-1, 1)),
            minmax_scale(vy, feature_range=(-1, 1)),
            minmax_scale(self.pressure, feature_range=(0, 1)),
            self.is_fake,
        )


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

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.pressure = pressure
        self.is_fake = is_fake