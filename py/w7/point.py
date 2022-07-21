class Point():
    def __init__(self, observation=None, observation_index=None, cluster_number=None):
        self._observation = observation
        self._cluster_number = cluster_number
        self._observation_index = observation_index
        self._n_dimensions = len(observation)

    def get_observation(self):
        return self._observation

    def get_cluster_number(self):
        return self._cluster_number

    def get_observation_index(self):
        return self._observation_index

    def get_n_dimensions(self):
        return self._n_dimensions

    def set_observation(self, observation=None):
        self._observation = observation

    def set_cluster_number(self, cluster_number=None):
        self._cluster_number = cluster_number

    def set_observation_index(self, observation_index=None):
        self._observation_index = observation_index
