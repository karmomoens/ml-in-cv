import numpy as np
import cv2


class FrameComponent:
    def get_image(self):
        raise NotImplementedError()

    def get_landmarks(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()


class FrameDecorator(FrameComponent):
    def get_parent(self):
        raise NotImplementedError()

    def set_parent(self, new_parent):
        raise NotImplementedError()


class NullComponent(FrameComponent):
    IMAGE = np.zeros([1, 1, 3], dtype=np.uint8)

    def get_image(self):
        return NullComponent.IMAGE

    def get_landmarks(self):
        return []

    def copy(self):
        return NullComponent()


class TrivialComponent(FrameComponent):
    def __init__(self):
        self.image = NullComponent.IMAGE
        self.landmarks = []

    def get_image(self):
        return self.image

    def get_landmarks(self):
        return self.landmarks

    def copy(self):
        return self


class LandmarkDetectingComponent(FrameComponent):
    def __init__(self, landmark_detector):
        self.image = NullComponent.IMAGE
        self.landmark_detector = landmark_detector

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def get_landmarks(self):
        return self.landmark_detector.get_all_landmarks(self.image)

    def copy(self):
        return self


class CachingLandmarkComponent(FrameComponent):
    def __init__(self, landmark_detector):
        self.detecting_component = LandmarkDetectingComponent(landmark_detector)
        self.landmarks = None

    def set_image(self, image):
        self.landmarks = None
        self.detecting_component.set_image(image)

    def get_image(self):
        return self.detecting_component.image

    def get_landmarks(self):
        if self.landmarks is None:
            self.landmarks = self.detecting_component.get_landmarks()
        return self.landmarks

    def copy(self):
        return self


class LandmarkOverlayDecorator(FrameDecorator):
    def __init__(self, parent_component=NullComponent()):
        self.parent = parent_component

    def get_parent(self):
        return self.parent

    def set_parent(self, new_parent):
        self.parent = new_parent

    def get_landmarks(self):
        return self.parent.get_landmarks()

    def get_image(self):
        result = self.parent.get_image()
        for landmark in self.get_landmarks():
            for i, (x, y) in enumerate(landmark):
                # if i != 21 and i != 22:
                #     continue
                cv2.circle(result, (x, y), 1, (0, 0, 255), -1)
        return result

    def copy(self):
        result = LandmarkOverlayDecorator()
        result.set_parent(self.parent.copy())
        return result
