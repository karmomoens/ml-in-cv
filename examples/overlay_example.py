import cv2

from src.machinelearningsuite.landmarkdetector import LandmarkDetector
from src.imageoverlay.framecomponent import CachingLandmarkComponent as LandmarkComponent
from src.imageoverlay.multifacesloopdecorator import MultiFacesLoopDecorator as AllFaces
from src.imageoverlay.configured_overlay_decorator import ConfiguredOverlayDecorator as SpriteDecorator


def overlay_example():
    detector_data_path = '../../data/shape_predictor_68_face_landmarks.dat'
    landmark_detector = LandmarkDetector(detector_data_path)
    base_component = LandmarkComponent(landmark_detector)
    hat = AllFaces.wrap(SpriteDecorator(base_component, '../sprites/Party_Hat'))
    glasses = AllFaces.wrap(SpriteDecorator(hat, '../sprites/sunglasses'))

    use_webcam = False
    video_stream = cv2.VideoCapture("../../videos/trump.mp4")
    if use_webcam:
        video_stream = cv2.VideoCapture(0)
    while True:
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        if use_webcam:
            frame = cv2.flip(frame, 1)
        base_component.set_image(frame)
        output = glasses.get_image()
        output = cv2.resize(output, (0, 0), fx=2.0, fy=2.0)
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    video_stream.release()


if __name__ == '__main__':
    overlay_example()
