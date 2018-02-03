from sklearn import preprocessing


class Normalizer:
    def __init__(self, configuration):
        self.configuration = configuration
        self.normalizer = preprocessing.Normalizer()

    def load_configuration(self):
        if self.configuration.normalizer:
            self.normalizer = self.configuration.normalizer

    def save_configuration(self):
        self.configuration.normalizer = self.normalizer

    def train(self):
        X = self.configuration.data_values
        self.normalizer.fit(X)
        self.configuration.data_values_normalized = self.normalizer.transform(X)
        self.save_configuration()

    def normalize(self, data):
        data_normalized = self.normalizer.transform(data)
        return data_normalized
