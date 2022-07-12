import numpy as np


class Cluster():
    def __init__(self, point_arr=None):
        self._point_arr = point_arr
        self._n_dimensions = self._compute_n_dimensions()
        self._center_point = self._compute_center_point()
        self._validate_point_arr()

    def _validate_point_arr(self):
        if type(self._point_arr) != np.ndarray:
            raise Exception(
                "Point array must be of type np.ndarray")

    def _compute_center_point(self):
        n_points = self.get_n_points()
        sum_arr = np.zeros(
            shape=(self._n_dimensions, ))
        for point in self._point_arr:
            observation = point.get_observation()
            sum_arr += observation
        center_point = sum_arr / n_points
        return center_point

    @staticmethod
    def _find_ndarray(searched=None, target=None):
        index = 0
        for array in searched:
            if np.array_equal(array, target) is True:
                return index
            index += 1
        return -1

    def _compute_n_dimensions(self):
        a_point = self._point_arr[0]
        return a_point.get_n_dimensions()

    def get_point_arr(self):
        return self._point_arr

    def get_n_points(self):
        return self._point_arr.shape[0]

    def get_center_point(self):
        return self._center_point

    def get_observation_index_arr(self):
        n_points = self.get_n_points()
        observation_index_arr = np.empty(
            shape=(n_points, ), dtype=np.int64)
        index_observation_index_arr = 0
        for point in self._point_arr:
            observation_index = point.get_observation_index()
            observation_index_arr[index_observation_index_arr] = observation_index
            index_observation_index_arr += 1
        return observation_index_arr

    def get_observation_arr(self, observation_set=None):
        observation_index_arr = self.get_observation_index_arr()
        n_points = self.get_n_points()
        observation_arr = np.empty(
            shape=(n_points, self._n_dimensions))
        index_observation_arr = 0
        for index in observation_index_arr:
            observation_arr[index_observation_arr] = observation_set[index]
            index_observation_arr += 1
        return observation_arr

    def add_point(self, point_obj=None):
        self._point_arr = np.append(
            self._point_arr, [point_obj], axis=0)
        self._center_point = self._compute_center_point()

    def remove_point(self, point_obj=None):
        index_to_delete = Cluster._find_ndarray(
            searched=self._point_arr, target=point_obj)
        if index_to_delete >= 0:
            self._point_arr = np.delete(
                self._point_arr, index_to_delete, axis=0)
        self._center_point = self._compute_center_point()
