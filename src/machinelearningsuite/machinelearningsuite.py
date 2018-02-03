import sys

import cv2
import numpy as np

from src.machinelearningsuite.configuration import Configuration
from src.machinelearningsuite.landmarkdetector import LandmarkDetector
from src.machinelearningsuite.featureprocessing import FeatureProcessing
from src.machinelearningsuite.normalizer import Normalizer
from src.machinelearningsuite.classifier import Classifier


class MachineLearningSuite:

    def __init__(self, source, predictor_file):
        self.configuration = Configuration()
        self.landmark_detector = LandmarkDetector(predictor_file=predictor_file)
        self.feature_processor = FeatureProcessing(self.configuration)
        self.normalizer = Normalizer(self.configuration)
        self.classifier = Classifier(self.configuration)
        self.source = cv2.VideoCapture(0) if source == "webcam" else cv2.VideoCapture(source)
        self.key_to_class = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

    def initialize(self):
        self.configuration.initialize()
        self.feature_processor.load_configuration()
        self.normalizer.load_configuration()
        self.classifier.load_configuration()

    def create_classes(self):
        print("1. Create classes")
        print("===================")
        print("Current classes: {}".format(self.configuration.classes))
        print("Press \"q\" to stop entering classes")
        i = 0
        while True:
            label = input("What is the name of class {} ?".format(i))
            if label == "q":
                break
            self.configuration.classes.append(label)
            i += 1
        self.configuration.save_configuration()

    def select_parts(self):
        print("1. Create classes")
        print("===================")
        print("Your selection has to be within the following list:")
        print([part for part in self.feature_processor.markers.keys()])
        print("Current face parts: {}".format(self.feature_processor.parts))
        print("Press \"q\" to stop entering face parts")
        while True:
            face_part = input("Enter a face part: ")
            if face_part == "q":
                break
            if face_part in self.feature_processor.markers:
                self.feature_processor.parts.append(face_part)
            else:
                print("Wrong face part!")
        self.feature_processor.save_configuration()
        self.configuration.save_configuration()

    def train(self):
        print("Press a to save feature vector for class 1, press b to save feature vector for class 2")
        while True:
            ret, frame = self.source.read()
            if type(frame) == type(None):
                break
            frame, landmarks = self.landmark_detector.get_frame_with_landmarks(frame)
            # feature_vector = self.feature_processor.process(landmarks, ["mouth", "right-eye", "left-eye", "left-eyebrow", "right-eyebrow"])
            feature_vector = self.feature_processor.process(landmarks)
            print(len(feature_vector) / 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if feature_vector:
                if chr(key) in self.key_to_class:
                    self.configuration.set_data_values(self.key_to_class[chr(key)], feature_vector)
                    print("You pressed on \"{}\": Feature vector saved!".format(chr(key)))
                else:
                    if key != 255:
                        print("This key is not set for training purposes")
            else:
                print("No landmarks detected!")
            if key == ord("q"):
                self.normalizer.train()
                self.classifier.train()
                self.configuration.save_configuration()
                break
        cv2.destroyAllWindows()

    def predict(self):
        while True:
            ret, frame = self.source.read()
            if type(frame) == type(None):
                break
            frame, landmarks = self.landmark_detector.get_frame_with_landmarks(frame)
            # feature_vector = self.feature_processor.process(landmarks, ["mouth", "right-eye", "left-eye", "left-eyebrow", "right-eyebrow"])
            feature_vector = self.feature_processor.process(landmarks)
            print(len(feature_vector)/2)
            cv2.imshow("Frame", frame)
            if feature_vector:
                feature_vector = np.asarray(feature_vector).reshape(1, -1)
                feature_vector_normalized = self.normalizer.normalize(feature_vector)
                print(np.asarray(feature_vector_normalized).shape[1]/2)
                prediction = self.classifier.predict(feature_vector_normalized)
                try:
                    predicted_class = self.configuration.classes[int(prediction[0])]
                    label = predicted_class
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label, (100, 400), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except IndexError:
                    print("This class has no label yet (class index: {})".format(prediction[0]))
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    def quit(self):
        self.source.release()
        sys.exit()