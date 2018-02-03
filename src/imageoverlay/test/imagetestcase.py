from unittest import TestCase
import cv2
from numpy.testing import assert_almost_equal


class ImageTestCase(TestCase):
    def assertUnalteredImage(self, image):
        output_path = './output/' + self.id() + '.bmp'
        previous_image = cv2.imread(output_path)
        cv2.imwrite(output_path, image)
        if previous_image is None:
            self.assertTrue(True)
        else:
            assert_almost_equal(previous_image, image)
