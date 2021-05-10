from xml.dom import minidom
import numpy as np
from matplotlib.path import Path as PltPath
from skimage.draw import polygon2mask


class Box:
    def __init__(self, min: np.array, max: np.array):
        self.min = min
        self.max = max

        self.width = int(np.ceil(max[0] - min[0]))
        self.height = int(np.ceil(max[1] - min[1]))

    def as_slice(self):
        x0 = int(np.floor(self.min[0]))
        x1 = int(np.ceil(self.max[0]))
        y0 = int(np.floor(self.min[1]))
        y1 = int(np.ceil(self.max[1]))

        return slice(x0, x1), slice(y0, y1)

    def __str__(self):
        x, y = self.as_slice()
        return "[{}, {}, shape=({}, {})]".format(x, y, self.width, self.height)


class Path(PltPath):
    def __init__(self, id: str, vertices, codes=None, _interpolation_steps=1, closed=False, readonly=False):
        super(Path, self).__init__(vertices, codes, _interpolation_steps, closed, readonly)
        self.id = id

    def __init_subclass__(cls, **kwargs):
        cls.id = kwargs["id"]

    def bounding_box(self):
        return Box(self.vertices.min(0), self.vertices.max(0))

    def box_and_mask(self):
        min = self.vertices.min(0)
        max = self.vertices.max(0)

        box = Box(min, max)

        local_shape = (box.width, box.height)
        local_polygon = self.vertices - min

        mask = polygon2mask(local_shape, local_polygon)

        return box, mask


class SVG:
    def __init__(self, file_path):
        doc = minidom.parse(file_path)

        path_elements = doc.getElementsByTagName("path")

        paths = np.empty(len(path_elements), dtype=Path)

        for i in range(len(path_elements)):
            path = path_elements[i]
            d: str = path.getAttribute("d")
            id: str = path.getAttribute("id")

            points = []
            point = np.array([0, 0])
            is_x = True
            for item in d.split():
                try:
                    val = float(item)
                    if is_x:
                        point[0] = val
                    else:
                        point[1] = val
                        points.append(point)
                        point = np.array([0, 0])

                    is_x = not is_x
                except ValueError:
                    pass

            path = Path(id, np.array(points))
            paths[i] = path

        self.paths = paths

    def words(self, img):
        words = []

        for path in self.paths:
            box, mask = path.box_and_mask()
            x, y = box.as_slice()

            sub_image = img[y, x] | ~mask.T

            words.append(sub_image)

        return np.array(words)

    def __str__(self):
        return self.paths.__str__()
