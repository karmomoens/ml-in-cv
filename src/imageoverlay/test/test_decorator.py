import cv2

from src.imageoverlay.test.imagetestcase import ImageTestCase

from src.machinelearningsuite.landmarkdetector import LandmarkDetector
from src.imageoverlay.framecomponent import CachingLandmarkComponent, LandmarkOverlayDecorator
from src.imageoverlay.multifacesloopdecorator import MultiFacesLoopDecorator
from src.imageoverlay.attached_overlay_decorator import AttachedOverlayDecorator
from src.imageoverlay.classification_decorator import ClassificationDecorator, ConstantClassifier


class TestDecorator(ImageTestCase):
    @classmethod
    def setUpClass(cls):
        detector_data_path = '../data/shape_predictor_68_face_landmarks.dat'
        cls.landmark_detector = LandmarkDetector(detector_data_path)

    def setUp(self):
        self.backdrop = cv2.imread('../tests/test_backdrop.jpg')
        self.hat = AttachedOverlayDecorator.construct_from_graphic_path('../sprites/Party_Hat.png')
        self.base_component = CachingLandmarkComponent(TestDecorator.landmark_detector)
        self.base_component.set_image(self.backdrop)

    def test_overlay_landmarks(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        result = landmark_overlay.get_image()
        self.assertUnalteredImage(result)

    def test_move_sprite(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        self.hat.parent = landmark_overlay
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        result = self.hat.get_image()
        self.assertUnalteredImage(result)

    def test_angle_offset(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        self.hat.parent = landmark_overlay
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.angle_offset = 90
        result = self.hat.get_image()
        self.assertUnalteredImage(result)

    def test_perpendicular_offset(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        self.hat.parent = landmark_overlay
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.perpendicular_offset = -50
        result = self.hat.get_image()
        self.assertUnalteredImage(result)

    def test_parallel_offset(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        self.hat.parent = landmark_overlay
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.parallel_offset = 30.5
        self.expected_landmark_distance = 300.0
        result = self.hat.get_image()
        self.assertUnalteredImage(result)

    def test_scale(self):
        landmark_overlay = LandmarkOverlayDecorator(self.base_component)
        self.hat.parent = landmark_overlay
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.expected_landmark_distance = 600.0
        result = self.hat.get_image()
        self.assertUnalteredImage(result)

    def test_multifaces(self):
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.perpendicular_offset = -20
        self.hat.expected_landmark_distance = 300.0
        multifaces = MultiFacesLoopDecorator(self.base_component)
        self.hat.parent = multifaces.get_start_of_loop_component()
        multifaces.set_end_of_loop_component(self.hat)
        result = multifaces.get_image()
        self.assertUnalteredImage(result)

    def test_multifaces_wrap(self):
        self.hat.parent = self.base_component
        self.hat.sprite_anchors = ((25, 105), (60, 110))
        self.hat.anchored_landmark_indexes = (21, 22)
        self.hat.perpendicular_offset = -20
        self.hat.expected_landmark_distance = 300.0
        multifaces = MultiFacesLoopDecorator.wrap(self.hat)
        result = multifaces.get_image()
        self.assertUnalteredImage(result)

    def test_forgot_input(self):
        base_component = CachingLandmarkComponent(TestDecorator.landmark_detector)
        base_component.get_landmarks()

    def set_up_classification_decorator(self):
        self.classification_decorator = ClassificationDecorator(self.base_component)
        self.classifier = ConstantClassifier()
        first_class_decorator = self.hat
        second_class_decorator = LandmarkOverlayDecorator()
        self.classification_decorator.classifier = self.classifier
        self.classification_decorator.set_decorator_for_class(first_class_decorator, 0)
        self.classification_decorator.set_decorator_for_class(second_class_decorator, 1)

    def test_classification_first_class(self):
        self.set_up_classification_decorator()
        self.classifier.prediction = 0
        result = self.classification_decorator.get_image()
        self.assertUnalteredImage(result)

    def test_classification_second_class(self):
        self.set_up_classification_decorator()
        self.classifier.prediction = 1
        result = self.classification_decorator.get_image()
        self.assertUnalteredImage(result)

    def test_classification_nonsense_class(self):
        self.set_up_classification_decorator()
        self.classifier.prediction = "nonsense"
        result = self.classification_decorator.get_image()
        self.assertUnalteredImage(result)
