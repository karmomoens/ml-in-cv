import cv2
from src.imageoverlay.point import Point
from src.imageoverlay.sprite import Sprite

from src.imageoverlay.test.imagetestcase import ImageTestCase


class TestSprite(ImageTestCase):
    def setUp(self):
        self.backdrop = cv2.imread('../tests/test_backdrop.jpg')
        self.hat = Sprite.construct_from_graphic_path('../sprites/Party_Hat.png')

    def test_basic_overlay(self):
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_top_corner_out_of_bounds(self):
        self.hat.target_location = Point(-20, -20)
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_bottom_corner_out_of_bounds(self):
        backdrop_height, backdrop_width, _ = self.backdrop.shape
        hat_height, hat_width, _ = self.hat.base_image.shape
        x = backdrop_width - hat_width + 20
        y = backdrop_height - hat_height + 20
        self.hat.target_location = Point(x, y)
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_semi_transparent(self):
        self.hat.base_alpha /= 2.0
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_scale_up(self):
        self.hat.scale = 2.6
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_scale_down(self):
        self.hat.scale = 0.3
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_rotate_ccw(self):
        self.hat.rotation = 90
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_rotate_cw(self):
        self.hat.rotation = -60
        test = self.hat.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)

    def test_null_sprite(self):
        sprite = Sprite()
        test = sprite.overlay_on(self.backdrop)
        self.assertUnalteredImage(test)
