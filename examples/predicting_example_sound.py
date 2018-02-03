import cv2
import pygame

from src.imageoverlay.classification_decorator import ClassificationDecorator as ClassDecorator
from src.imageoverlay.framecomponent import CachingLandmarkComponent as LandmarkComponent
from src.imageoverlay.multifacesloopdecorator import MultiFacesLoopDecorator as AllFaces
from src.imageoverlay.framecomponent import LandmarkOverlayDecorator
from src.machinelearningsuite.landmarkdetector import LandmarkDetector
from src.machinelearningsuite.predictorinterface import PredictorInterface
from src.imageoverlay.configured_overlay_decorator import ConfiguredOverlayDecorator as SpriteDecorator
from src.imageoverlay.sounddecorator import SoundDecorator


def predicting_example():
    pygame.init()
    # Instanciate a new landmark detector
    detector_data_path = '../../data/shape_predictor_68_face_landmarks.dat'
    landmark_detector = LandmarkDetector(predictor_file=detector_data_path)

    # Create a frame component with landmarks
    base_component = LandmarkComponent(landmark_detector)

    # Instantiate and initialize the trained predictor
    predictor = PredictorInterface('../examples/mouth.pkl')
    predictor.initialize()

    landmarks = LandmarkOverlayDecorator(base_component)

    # Add decorator for the predictor
    predictor_decorator = ClassDecorator(parent_component=landmarks, classifier=predictor)
    sound = SoundDecorator(landmarks, "../sounds/boring.wav")
    predictor_decorator.set_decorator_for_class(sound, 0)
    multifaces = AllFaces.wrap(predictor_decorator)

    hat = AllFaces.wrap(SpriteDecorator(multifaces, '../sprites/Party_Hat'))

    use_webcam = True
    video_stream = cv2.VideoCapture("../../videos/trump.mp4")
    if use_webcam:
        video_stream = cv2.VideoCapture(0)
    while True:
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        if use_webcam:
            frame = cv2.flip(frame, 1)
        base_component.set_image(frame)
        output = hat.get_image()
        output = cv2.resize(output, (0, 0), fx=1.0, fy=1.0)
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    video_stream.release()


if __name__ == '__main__':
    predicting_example()
