from time import time
from pygame.mixer import Sound

from src.imageoverlay.framecomponent import FrameDecorator, NullComponent


class SoundDecorator(FrameDecorator):
    def __init__(self, parent_component=NullComponent(), sound_file_path=''):
        self.sound_file_path = sound_file_path
        self.parent = parent_component
        self.sound = Sound(sound_file_path)
        self.start_time = 0

    def get_parent(self):
        return self.parent

    def set_parent(self, new_parent):
        self.parent = new_parent

    def get_landmarks(self):
        self.parent.get_landmarks()

    def get_image(self):
        self.play()
        return self.parent.get_image()

    def play(self):
        if not self.is_playing():
            self.start_time = time()
            self.sound.play()

    def is_playing(self):
        now = time()
        return self.sound.get_length() > now - self.start_time

    def stop(self):
        self.sound.stop()
        self.start_time = 0

    def copy(self):
        return SoundDecorator(parent_component=self.parent.copy(), sound_file_path=self.sound_file_path)
