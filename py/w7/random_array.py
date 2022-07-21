import math
import random

import numpy as np


class RandomArray():

    def __init__(self, range_=None, len_=None):
        self._range_ = range_
        self._len_ = len_

    def get_random_array(self):
        random_array = np.zeros(
            shape=(self._len_,), dtype="int64")
        for i in range(self._len_):
            random_array[i] = random.randint(
                *self._range_)
        return random_array
