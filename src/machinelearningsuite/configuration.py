import pickle


class Configuration:

    def __init__(self, configuration_file='config.pkl'):
        self.classes = []
        self.normalizer = None
        self.classifier = None
        self.feature_processor = None
        self.data_labels = []
        self.data_values = []
        self.data_values_normalized = []
        self.configuration_file = configuration_file

    def initialize(self):
        print("Current configuration:")
        print("-----------------------")
        try:
            configuration_dict = self.load_configuration()
            self.from_dict(configuration_dict)
            print(configuration_dict)
        except:
            print("No configuration yet")
        print("\n")

    def load_configuration(self):
        with open(self.configuration_file, 'rb') as f:
            return pickle.load(f)

    def save_configuration(self):
        configuration_dict = self.to_dict()
        with open(self.configuration_file, 'wb') as f:
            pickle.dump(configuration_dict, f, pickle.HIGHEST_PROTOCOL)

    def reset(self):
        self.classes = []
        self.normalizer = None
        self.classifier = None
        self.feature_processor = []
        self.data_labels = []
        self.data_values = []
        self.data_values_normalized = []

    def from_dict(self, config_dict):
        self.classes = config_dict.get("classes")
        self.normalizer = config_dict.get("normalizer")
        self.classifier = config_dict.get("classifier")
        self.feature_processor = config_dict.get("feature-processor")
        self.data_labels = config_dict.get("data-labels")
        self.data_values = config_dict.get("data-values")
        self.data_values_normalized = config_dict.get("data-values-normalized")

    def to_dict(self):
        return {"classes": self.classes,
                "normalizer": self.normalizer,
                "classifier": self.classifier,
                "feature-processor": self.feature_processor,
                "data-labels": self.data_labels,
                "data-values": self.data_values,
                "data-values-normalized": self.data_values_normalized}

    def set_data_values(self, label, data):
        self.data_labels.append(label)
        self.data_values.append(data)
