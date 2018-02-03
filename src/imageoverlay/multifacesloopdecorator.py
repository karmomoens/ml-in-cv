import numpy as np

from src.imageoverlay.framecomponent import FrameDecorator, NullComponent, TrivialComponent


class MultiFacesLoopDecorator(FrameDecorator):
    def __init__(self, parent_component=NullComponent()):
        self.parent = parent_component
        self.start_of_loop_component = TrivialComponent()
        self.end_of_loop_components = [self.start_of_loop_component]

    def get_parent(self):
        return self.parent

    def set_parent(self, new_parent):
        self.parent = new_parent

    def get_start_of_loop_component(self):
        return self.start_of_loop_component

    def set_end_of_loop_component(self, component):
        for i, _ in enumerate(self.end_of_loop_components):
            self.end_of_loop_components[i] = component.copy()

    def add_to_loop(self, decorator):
        for i, component in enumerate(self.end_of_loop_components):
            new_decorator = decorator.copy()
            new_decorator.set_parent(component)
            self.end_of_loop_components[i] = new_decorator

    @staticmethod
    def wrap(decorator):
        result = MultiFacesLoopDecorator(decorator.get_parent())
        result.add_to_loop(decorator)
        return result

    def get_landmarks(self):
        return self.parent.get_landmarks()

    def copy(self):
        result = MultiFacesLoopDecorator(self.parent.copy())
        # start of loop needs to be the same because all decorators are attached to it
        # shouldn't hurt because its properties are reset in every loop
        result.start_of_loop_component = self.start_of_loop_component
        result.end_of_loop_components = [decorator.copy() for decorator in self.end_of_loop_components]
        return result

    def get_image(self):
        image_buffer = self.parent.get_image()
        all_faces = self.parent.get_landmarks()
        self.provide_separate_decorators_for_new_faces()
        for face_id, face in enumerate(all_faces):
            self.start_of_loop_component.image = image_buffer
            self.start_of_loop_component.landmarks = [face]
            image_buffer = self.end_of_loop_components[face_id].get_image()
        return image_buffer

    def provide_separate_decorators_for_new_faces(self):
        all_faces = self.parent.get_landmarks()
        for _ in range(0, len(all_faces) - len(self.end_of_loop_components), 1):
            self.end_of_loop_components.append(self.end_of_loop_components[0].copy())

