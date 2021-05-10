from xml.dom import minidom
import numpy as np


class Box:
    def __init__(self, min: np.array, max: np.array):
        self.min = min
        self.max = max


class Path:
    def __init__(self, id: str, points: np.array):
        self.id = id
        self.points = points

    def bounding_box(self):
        return Box(self.points.min(), self.points.max())

    def __str__(self):
        return "{} {}".format(self.id, self.points)


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

                    is_x = not is_x
                except ValueError:
                    pass

            path = Path(id, np.array(points))
            paths[i] = path

        self.paths = paths

    def __str__(self):
        return self.paths.__str__()
