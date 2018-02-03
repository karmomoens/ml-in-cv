import cv2
import numpy as np

from src.imageoverlay.point import Point


class Sprite:
    def __init__(self):
        self.base_image = np.zeros([1, 1, 3])
        self.base_alpha = np.zeros([1, 1, 3])
        self.scale = 1.0
        self.rotation = 0.0
        self.target_location = Point(0, 0)
        self.center_of_transformations = Point(0, 0)

    @staticmethod
    def construct_from_graphic_path(path):
        result = Sprite()
        graphic = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        result.base_image = graphic[:, :, :3]
        single_alpha_channel = graphic[:, :, 3].astype(float)/255.0
        result.base_alpha = np.repeat(single_alpha_channel[:, :, None], 3, axis=2)
        return result

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def overlay_on(self, frame):
        float_frame = frame.astype(float)
        float_frame[self.get_valid_backdrop_slices_for_shape(frame.shape)] = \
            cv2.multiply(1.0 - self.alpha_channel[self.get_valid_sprite_slices_for_shape(frame.shape)],
                         float_frame[self.get_valid_backdrop_slices_for_shape(frame.shape)])
        float_frame[self.get_valid_backdrop_slices_for_shape(frame.shape)] += \
            cv2.multiply(self.alpha_channel, self.image.astype(float))[self.get_valid_sprite_slices_for_shape(frame.shape)]
        return float_frame.astype(np.uint8)

    def get_valid_backdrop_slices_for_shape(self, shape):
        first_overlaid = self.calculate_first_overlaid_point()
        last_overlaid = self.calculate_last_overlaid_point(shape)
        return [slice(first_overlaid.y, last_overlaid.y), slice(first_overlaid.x, last_overlaid.x)]

    def get_valid_sprite_slices_for_shape(self, shape):
        first_sprite = self.calculate_first_visible_sprite_pixel()
        last_sprite = self.calculate_last_visible_sprite_pixel(shape)
        return [slice(first_sprite.y, last_sprite.y), slice(first_sprite.x, last_sprite.x)]

    @property
    def image(self):
        return self.sprite_tranformer.transform(self.base_image)

    @property
    def alpha_channel(self):
        return self.sprite_tranformer.transform(self.base_alpha)

    @property
    def sprite_tranformer(self):
        transformer = BoundAdjustingImageTransformer()
        transformer.input_angle = self.rotation
        transformer.input_scale = self.scale
        return transformer

    def get_point_after_transformations(self, base_point):
        transformer = self.sprite_tranformer
        transformer.input_image = self.base_image
        return transformer.get_point_after_transformations(base_point)

    def calculate_first_visible_sprite_pixel(self):
        y = max(-self.translation_vector.y, 0)
        x = max(-self.translation_vector.x, 0)
        return Point(x, y)

    def calculate_last_visible_sprite_pixel(self, shape):
        backdrop_height, backdrop_width, _ = shape
        out_of_bounds_y = max(0, (self.translation_vector.y+self.height) - backdrop_height)
        y = self.height - out_of_bounds_y
        out_of_bounds_x = max(0, (self.translation_vector.x + self.width) - backdrop_width)
        x = self.width - out_of_bounds_x
        return Point(x, y)

    def calculate_first_overlaid_point(self):
        y = max(0, self.translation_vector.y)
        x = max(0, self.translation_vector.x)
        return Point(x, y)

    def calculate_last_overlaid_point(self, shape):
        actual_overlaid_height, actual_overlaid_width = self.calculate_actual_overlaid_shape(shape)
        first_overlaid = self.calculate_first_overlaid_point()
        y = first_overlaid.y + actual_overlaid_height
        x = first_overlaid.x + actual_overlaid_width
        return Point(x, y)

    def calculate_actual_overlaid_shape(self, shape):
        backdrop_height, backdrop_width, _ = shape
        first_sprite = self.calculate_first_visible_sprite_pixel()
        last_sprite = self.calculate_last_visible_sprite_pixel(shape)
        actual_overlaid_height = last_sprite.y - first_sprite.y
        actual_overlaid_width = last_sprite.x - first_sprite.x
        return actual_overlaid_height, actual_overlaid_width

    @property
    def translation_vector(self):
        center_after_transformation = self.get_point_after_transformations(self.center_of_transformations)
        return self.target_location - center_after_transformation


class BoundAdjustingImageTransformer:
    def __init__(self):
        self.input_image = np.zeros([0, 0, 3])
        self.input_angle = 0.0
        self.input_scale = 1.0
        self.output_bounds = Point(0, 0)

    @property
    def input_bounds(self):
        (height, width) = self.input_image.shape[:2]
        return Point(width, height)

    @property
    def input_center(self):
        return Point(self.input_bounds.x // 2, self.input_bounds.y // 2)

    @property
    def unadjusted_transformation_matrix(self):
        return cv2.getRotationMatrix2D((self.input_center.x, self.input_center.y), -self.input_angle, self.input_scale)

    def calculate_output_bounds(self):
        rotation_matrix = self.unadjusted_transformation_matrix
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        width_new = int((self.input_bounds.y * sin) + (self.input_bounds.x * cos))
        height_new = int((self.input_bounds.y * cos) + (self.input_bounds.x * sin))
        self.output_bounds = Point(width_new, height_new)

    def transform(self, image):
        self.input_image = image
        self.calculate_output_bounds()
        rotation_matrix = self.get_adjusted_tranformation_matrix()
        return cv2.warpAffine(self.input_image, rotation_matrix, (self.output_bounds.x, self.output_bounds.y))

    def get_adjusted_tranformation_matrix(self):
        rotation_matrix = self.unadjusted_transformation_matrix
        rotation_matrix[0, 2] += (self.output_bounds.x / 2) - self.input_center.x
        rotation_matrix[1, 2] += (self.output_bounds.y / 2) - self.input_center.y
        return rotation_matrix

    def get_point_after_transformations(self, base_point):
        self.calculate_output_bounds()
        rotation_matrix = self.get_adjusted_tranformation_matrix()
        real_point = base_point.clone().apply_rotation_matrix(rotation_matrix)
        return Point(int(real_point.x), int(real_point.y))
