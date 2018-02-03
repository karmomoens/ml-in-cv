import collections

import numpy as np


class Filter:

    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.queue = collections.deque(maxlen=queue_size)

    def filter(self, prediction):
        self.queue.append(prediction[0])
        if len(self.queue) == self.queue_size:
            queue = np.asarray(self.queue)
            return np.bincount(queue).argmax()
        else:
            return None
