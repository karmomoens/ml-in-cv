import json

from src.imageoverlay.framecomponent import FrameDecorator, NullComponent
from src.imageoverlay.attached_overlay_decorator import AttachedOverlayDecorator


class ConfiguredOverlayDecorator(FrameDecorator):
    GRAPHICS_EXTENSION = ".png"
    CONFIGURATION_EXTENSION = ".json"

    def __init__(self, parent_component=NullComponent(), base_file_name=""):
        self.base_file_name = base_file_name
        self.attached_decorator = AttachedOverlayDecorator.construct_from_graphic_path(self.graphics_path)
        self.attached_decorator.parent = parent_component
        self.load_configuration()

    def get_parent(self):
        return self.attached_decorator.get_parent()

    def set_parent(self, value):
        self.attached_decorator.set_parent(value)

    @property
    def graphics_path(self):
        return self.base_file_name + ConfiguredOverlayDecorator.GRAPHICS_EXTENSION

    @property
    def config_path(self):
        return self.base_file_name + ConfiguredOverlayDecorator.CONFIGURATION_EXTENSION

    def load_configuration(self):
        config = self.read_config_file()
        self.apply_configuration(config)

    def read_config_file(self):
        data_dict = json.load(open(self.config_path))
        return data_dict

    def apply_configuration(self, config_dict):
        self.attached_decorator.sprite_anchors = config_dict.get("sprite_anchors", ((-1, 0), (1, 0)))
        self.attached_decorator.anchored_landmark_indexes = config_dict.get("anchored_landmark_indexes", (0, 1))
        self.attached_decorator.perpendicular_offset = config_dict.get("perpendicular_offset", 0.0)
        self.attached_decorator.parallel_offset = config_dict.get("parallel_offset", 0.0)
        self.attached_decorator.angle_offset = config_dict.get("angle_offset", 0.0)
        self.attached_decorator.scale_factor = config_dict.get("scale_factor", None)

    def get_image(self):
        return self.attached_decorator.get_image()

    def get_landmarks(self):
        return self.attached_decorator.get_landmarks()

    def copy(self):
        return ConfiguredOverlayDecorator(self.get_parent().copy(), self.base_file_name)
