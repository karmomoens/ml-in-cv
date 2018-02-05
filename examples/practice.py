import cv2

from src.machinelearningsuite.landmarkdetector import LandmarkDetector
from src.imageoverlay.framecomponent import CachingLandmarkComponent as LandmarkComponent
from src.imageoverlay.multifacesloopdecorator import MultiFacesLoopDecorator as AllFaces
from src.imageoverlay.configured_overlay_decorator import ConfiguredOverlayDecorator as SpriteDecorator
from src.machinelearningsuite.predictorinterface import PredictorInterface
from src.imageoverlay.classification_decorator import ClassificationDecorator as ClassDecorator


def practice():
    video_stream = cv2.VideoCapture("../../videos/trump.mp4")
    while True:
        ret, frame = video_stream.read()
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        output = frame
        output = cv2.resize(output, (0, 0), fx=2.0, fy=2.0)
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    video_stream.release()


if __name__ == '__main__':
    practice()
