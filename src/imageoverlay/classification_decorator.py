from src.imageoverlay.framecomponent import FrameDecorator, NullComponent


class NullClassifier:
    def predict(self, landmarks):
        return 0

    def copy(self):
        return NullClassifier()


class ClassificationDecorator(FrameDecorator):
    def __init__(self, parent_component=NullComponent(), classifier=NullClassifier()):
        self.parent = parent_component
        self.classifier = classifier
        self.decorators = dict()

    def set_decorator_for_class(self, decorator, classification):
        decorator.set_parent(self.parent)
        self.decorators[classification] = decorator

    def get_landmarks(self):
        return self.parent.get_landmarks()

    def get_image(self):
        prediction = self.classifier.predict(self.get_landmarks())
        selected_decorator = self.decorators.get(prediction, self.parent)
        return selected_decorator.get_image()

    def get_parent(self):
        return self.parent

    def set_parent(self, new_parent):
        self.parent = new_parent
        all_decorators = self.decorators.values()
        for decorator in all_decorators:
            decorator.set_parent(new_parent)

    def copy(self):
        result = ClassificationDecorator(self.parent.copy(), self.classifier.copy())
        for key, value in self.decorators.items():
            result.set_decorator_for_class(value.copy(), key)
        return result


class ConstantClassifier:
    def __init__(self, constant_prediction=0):
        self.prediction = constant_prediction

    def predict(self, landmarks):
        return self.prediction

    def copyt(self):
        return ConstantClassifier(self.prediction)
