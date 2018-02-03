from sklearn import svm


class Classifier:
    def __init__(self, configuration):
        self.configuration = configuration
        self.classifier = svm.SVC(kernel='linear', C=1000)
        # self.classifier = svm.SVC()
        # self.classifier = svm.LinearSVC()
        # self.classifier = svm.SVC(decision_function_shape='ovo')

    def load_configuration(self):
        if self.configuration.classifier:
            self.classifier = self.configuration.classifier

    def save_configuration(self):
        self.configuration.classifier = self.classifier

    def train(self):
        X = self.configuration.data_values_normalized
        y = self.configuration.data_labels
        self.classifier.fit(X, y)
        self.save_configuration()

    def predict(self, data):
        return self.classifier.predict(data)

