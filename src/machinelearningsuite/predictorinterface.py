import collections

import numpy as np

from src.machinelearningsuite.featureprocessing import FeatureProcessing
from src.machinelearningsuite.normalizer import Normalizer
from src.machinelearningsuite.classifier import Classifier
from src.machinelearningsuite.configuration import Configuration
from src.machinelearningsuite.featureprocessing import FeatureProcessing
from src.machinelearningsuite.filter import Filter


class PredictorInterface:
    def __init__(self, configuration_file):
        self.configuration_file = configuration_file
        self.configuration = Configuration(configuration_file=configuration_file)
        self.feature_processor = FeatureProcessing(self.configuration)
        self.normalizer = Normalizer(self.configuration)
        self.classifier = Classifier(self.configuration)
        self.filter = Filter(queue_size=10)
        self.initialize()

    def initialize(self):
        self.configuration.initialize()
        self.feature_processor.load_configuration()
        self.normalizer.load_configuration()
        self.classifier.load_configuration()

    def predict(self, landmarks):
        feature_vector = self.feature_processor.process(landmarks)
        if feature_vector:
            normalized_feature_vector = self.normalizer.normalize(np.asarray(feature_vector).reshape(1, -1))
            prediction = self.classifier.predict(normalized_feature_vector)
            return self.filter.filter(prediction)
        else:
            return None

    def copy(self):
        result = PredictorInterface(self.configuration_file)
        return result
