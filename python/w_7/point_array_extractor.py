import numpy as np

import point


class PointArrayExtractor():
    @staticmethod
    def extract_point_arr(observation_set=None):
        n_points = observation_set.shape[0]
        point_obj_arr = np.empty(
            shape=(n_points,), dtype=object)
        index_to_add_point_obj = 0
        for index, observation in enumerate(observation_set):
            point_obj = point.Point(
                observation=observation, observation_index=index)
            point_obj_arr[index_to_add_point_obj] = point_obj
            index_to_add_point_obj += 1
        return point_obj_arr
