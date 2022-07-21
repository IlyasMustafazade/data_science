import numpy as np

import kmeans


class Elbow():
    def __init__(self, kmeans_obj=None, n_cluster_range=None):
        self._n_cluster_range = range(
            n_cluster_range[0], n_cluster_range[1])
        self._kmeans_obj = kmeans_obj
        self._n_cluster_inertia_dict = {}

    def fit(self):
        for n_cluster in self._n_cluster_range:
            self._kmeans_obj.set_n_clusters(
                n_clusters=n_cluster)
            self._kmeans_obj.fit()
            self._n_cluster_inertia_dict[n_cluster] = self._kmeans_obj.inertia_

    def get_n_cluster_arr(self):
        n_cluster_arr = list(
            self._n_cluster_inertia_dict.keys())
        n_cluster_arr = np.array(n_cluster_arr)
        return n_cluster_arr

    def get_inertia_arr(self):
        inertia_arr = list(
            self._n_cluster_inertia_dict.values())
        inertia_arr = np.array(inertia_arr)
        return inertia_arr
