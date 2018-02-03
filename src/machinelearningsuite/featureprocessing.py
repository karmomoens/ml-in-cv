import numpy as np


class FeatureProcessing:

    def __init__(self, configuration):
        self.configuration = configuration
        self.parts = []
        self.markers = {
            "left-eyebrow": {
                "idx": list(range(17, 22)),
                "ref": 19
            },
            "right-eyebrow": {
                "idx": list(range(22, 27)),
                "ref": 24
            },
            "left-eye": {
                "idx": list(range(36, 42)),
                "ref": 41
            },
            "right-eye": {
                "idx": list(range(42, 48)),
                "ref": 46
            },
            "nose": {
                "idx": list(range(27, 36)),
                "ref": 30
            },
            "mouth": {
                "idx": list(range(48, 68)),
                "ref": 57
            },
            "jaw": {
                "idx": list(range(0, 17)),
                "ref": 8
            }
        }

    def load_configuration(self):
        if self.configuration.feature_processor:
            self.parts = self.configuration.feature_processor.parts

    def save_configuration(self):
        self.configuration.feature_processor = self

    def process(self, landmarks):
        if landmarks:
            if not len(landmarks[0]) == 68:
                return []
        else:
            return []

        for part in self.parts:
            if not part in self.markers:
                print("{} is not accepted as a face part".format(part))
                return []

        raw_feature_vector = []
        for part in self.parts:
            ref = self.markers[part]["ref"]
            markers = np.asarray(self.markers[part]["idx"])

            part_landmarks = [landmarks[0][i] for i in markers]
            ref_landmark = landmarks[0][ref]

            for landmark in part_landmarks:
                landmark_diff = np.subtract(landmark, ref_landmark)
                for element in landmark_diff:
                    raw_feature_vector.append(element)
        return raw_feature_vector

    def __str__(self):
        return "FeatureProcessing(parts={})".format(self.parts)

    def __repr__(self):
        return "FeatureProcessing(parts={})".format(self.parts)
