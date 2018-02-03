import cv2
import dlib
from imutils import face_utils


class LandmarkDetector:

    def __init__(self, predictor_file="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_file)

    def get_all_landmarks(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        landmarks = []
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks.append(shape)
        return landmarks

    def get_frame_with_landmarks(self, frame):
        frame_with_landmarks = frame.copy()
        landmarks = self.get_all_landmarks(frame)

        for landmark in landmarks:
            for (x, y) in landmark:
                cv2.circle(frame_with_landmarks, (x, y), 1, (0, 0, 255), -1)

        return frame_with_landmarks, landmarks
