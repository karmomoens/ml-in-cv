import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if self.x.__lt__(other.x):
            return True
        elif self.x.__eq__(other.x):
            return self.y.__lt__(other.y)

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y)

    @staticmethod
    def calculate_center_of_mass_coordinates(points):
        x = sum([elem.x for elem in points]) / len(points)
        y = sum([elem.y for elem in points]) / len(points)
        return Point(x, y)

    def translate_by_vector(self, vector_end_point):
        self.x += vector_end_point.x
        self.y += vector_end_point.y
        return self

    def as_cv_point(self):
        return self.x, self.y

    def apply_rotation_matrix(self, rotation_matrix):
        vector = self.to_rotatable_vector()
        transformed = np.dot(rotation_matrix, vector)
        transformed_point = Point.from_rotatable_vector(transformed)
        self.x = transformed_point.x
        self.y = transformed_point.y
        return self

    def to_rotatable_vector(self):
        vector = np.ones([3, 1])
        vector[0, 0] = self.x
        vector[1, 0] = self.y
        return vector

    @staticmethod
    def from_rotatable_vector(vector):
        return Point(vector[0, 0], vector[1, 0])

    def mirror(self, origin_point):
        distance_to_origin = self.x - origin_point.x
        self.x = origin_point.x - distance_to_origin
        return self

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        return self

    def clone(self):
        return Point(self.x, self.y)
