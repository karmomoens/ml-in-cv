from math import atan, pi, degrees, sin, cos, sqrt

import numpy as np
from src.imageoverlay.framecomponent import FrameDecorator, NullComponent
from src.imageoverlay.point import Point

from src.imageoverlay.sprite import Sprite


class AttachedOverlayDecorator(FrameDecorator):
    def __init__(self, parent_component=NullComponent, sprite=Sprite()):
        self.parent = parent_component
        self.sprite = sprite
        self.sprite_anchors = ((-1, 0), (1, 0))
        self.anchored_landmark_indexes = (0, 1)
        self.perpendicular_offset = 0.0
        self.parallel_offset = 0.0
        self.angle_offset = 0.0
        self.scale_factor = None

    @staticmethod
    def construct_from_graphic_path(path):
        sprite = Sprite.construct_from_graphic_path(path)
        return AttachedOverlayDecorator(sprite=sprite)

    def get_parent(self):
        return self.parent

    def set_parent(self, new_parent):
        self.parent = new_parent

    def get_landmarks(self):
        return self.parent.get_landmarks()

    def has_detected_landmarks(self):
        return 0 < len(self.get_landmarks())

    def get_image(self):
        if not self.has_detected_landmarks():
            return self.parent.get_image()
        self.rotate_sprite()
        self.move_sprite()
        self.scale_sprite()
        return self.sprite.overlay_on(self.parent.get_image())

    def rotate_sprite(self):
        self.sprite.rotation = degrees(self.calculate_required_sprite_rotation())
        self.sprite.rotation += self.angle_offset

    def scale_sprite(self):
        if self.scale_factor is None:
            self.sprite.scale = 1.0
        else:
            landarks_distance = self.calculate_anchored_landmark_distance()
            anchor_distance = self.calculate_sprite_anchor_distance()
            self.sprite.scale = self.scale_factor * landarks_distance / anchor_distance

    def move_sprite(self):
        landmark_center = self.calculate_anchored_landmark_center()
        self.sprite.target_location = Point(landmark_center[0], landmark_center[1])
        self.sprite.center_of_transformations = Point(self.calculate_sprite_anchor_center()[0], self.calculate_sprite_anchor_center()[1])
        self.move_perpendicular()
        self.move_parallel()

    def move_perpendicular(self):
        slope = self.calculate_anchored_landmark_slope()
        self.sprite.target_location.x += -1 * int(self.perpendicular_offset * sin(slope))
        self.sprite.target_location.y += int(self.perpendicular_offset * cos(slope))

    def move_parallel(self):
        slope = self.calculate_anchored_landmark_slope()
        self.sprite.target_location.x += int(self.parallel_offset * cos(slope))
        self.sprite.target_location.y += int(self.parallel_offset * sin(slope))

    def calculate_anchored_landmark_center(self):
        ldmk_0, ldmk_1 = self.get_anchored_landmarks()
        return calculate_two_point_center(ldmk_0, ldmk_1)

    def get_anchored_landmarks(self):
        all_landmarks = self.get_landmarks()[0]
        ldmk_0 = all_landmarks[self.anchored_landmark_indexes[0]]
        ldmk_1 = all_landmarks[self.anchored_landmark_indexes[1]]
        return ldmk_0, ldmk_1

    def calculate_sprite_anchor_center(self):
        return calculate_two_point_center(self.sprite_anchors[0], self.sprite_anchors[1])

    def calculate_required_sprite_rotation(self):
        landmark_slope = self.calculate_anchored_landmark_slope()
        anchor_slope = self.calculate_sprite_anchor_slope()
        return landmark_slope - anchor_slope

    def calculate_anchored_landmark_slope(self):
        ldmk_0, ldmk_1 = self.get_anchored_landmarks()
        return calculate_two_point_slope(ldmk_0, ldmk_1)

    def calculate_sprite_anchor_slope(self):
        return calculate_two_point_slope(self.sprite_anchors[0], self.sprite_anchors[1])

    def calculate_sprite_anchor_distance(self):
        return calculate_two_point_distance(self.sprite_anchors[0], self.sprite_anchors[1])

    def calculate_anchored_landmark_distance(self):
        ldmk_0, ldmk_1 = self.get_anchored_landmarks()
        return calculate_two_point_distance(ldmk_0, ldmk_1)

    def copy(self):
        result = AttachedOverlayDecorator(self.parent.copy(), self.sprite)
        result.sprite_anchors = self.sprite_anchors
        result.anchored_landmark_indexes = self.anchored_landmark_indexes
        result.perpendicular_offset = self.perpendicular_offset
        result.parallel_offset = self.parallel_offset
        result.angle_offset = self.angle_offset
        result.scale_factor = self.scale_factor
        return result


def calculate_two_point_distance(point_0, point_1):
    diff_x = point_1[0] - point_0[0]
    diff_y = point_1[1] - point_0[1]
    return sqrt(diff_x*diff_x + diff_y*diff_y)


def calculate_two_point_center(point_0, point_1):
    return ((point_0[0] + point_1[0]) // 2, (point_0[1] + point_1[1]) // 2)


def calculate_two_point_slope(point_0, point_1):
    slope_numerator = point_1[1] - point_0[1]
    slope_denominator = point_1[0] - point_0[0]
    return division_by_0_safe_atan(slope_numerator, slope_denominator)


def division_by_0_safe_atan(numerator, denominator):
    if denominator != 0:
        line_slope = numerator / denominator
        return atan(line_slope)
    else:
        return np.sign(numerator) * pi / 2.0
