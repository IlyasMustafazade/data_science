import numpy as np

import base_estimator
import cluster
import distance
import distance_chooser
import euclidean_distance
import point
import point_array_extractor
import random_array
import rectilinear_distance


class KMeans(base_estimator.BaseEstimator):
    def __init__(
            self,
            observation_set=None,
            n_clusters=8,
            distance_metric="euclidean",
            max_iter=300,
            verbose=False
    ):
        self._observation_set = observation_set
        self._validate_observation_set()
        self._n_clusters = n_clusters
        self._observation_set_shape = np.shape(
            self._observation_set)
        self._n_points = self._observation_set_shape[0]
        self._validate_n_clusters()
        self._n_dimensions = self._observation_set_shape[1]
        self._verbose = verbose
        self._distance_metric = distance_chooser.DistanceChooser.choose_distance_metric(
            distance_metric_as_str=distance_metric)
        self._cluster_dict = {}
        self._max_iter = max_iter
        point_extractor = point_array_extractor.PointArrayExtractor()
        self._point_obj_arr = point_extractor.extract_point_arr(
            self._observation_set)
        self.inertia_ = 0
        self.labels_ = None

    def _validate_observation_set(self):
        desired_dim = 2
        actual_dim = self._observation_set.ndim
        if desired_dim != actual_dim:
            raise Exception(
                "Observation set must be " + str(desired_dim) + "-dimensional")

    def _validate_n_clusters(self):
        if self._n_clusters >= self._n_points:
            raise Exception(
                "Number of clusters must be less than number of points")

    def _choose_initial_centers(self):
        random_array_obj = random_array.RandomArray(
            range_=(0, self._n_points-1), len_=self._n_clusters)
        random_arr = random_array_obj.get_random_array()
        center_dict = {}
        key = 0
        for i in range(self._n_clusters):
            random_index = random_arr[i]
            random_observation = self._observation_set[random_index]
            random_point = point.Point(observation=random_observation,
                                       observation_index=random_index,
                                       cluster_number=key)
            point_arr = [random_point]
            point_arr = np.array(point_arr)
            center_dict[key] = cluster.Cluster(
                point_arr=point_arr)
            key += 1
        return center_dict

    def _compute_closest_cluster_index(self, point_obj=None, cluster_dict=None):
        observation = point_obj.get_observation()
        dist_arr = np.zeros(
            shape=(self._n_clusters, ))
        for cluster_number, cluster_obj in cluster_dict.items():
            cluster_center_point = cluster_obj.get_center_point()
            dist_obj = self._distance_metric(
                object_a=cluster_center_point, object_b=observation)
            dist = dist_obj.get_distance()
            dist_arr[cluster_number] = dist
        min_val = np.min(dist_arr)
        min_index = np.where(dist_arr == min_val)
        min_index = min_index[0][0]
        return min_index

    def _set_inertia(self):
        inertia = 0
        cluster_arr = list(
            self._cluster_dict.values())
        cluster_arr = np.array(cluster_arr)
        for cluster_obj in cluster_arr:
            cluster_inertia = cluster_obj.get_inertia()
            inertia += cluster_inertia
        self.inertia_ = inertia

    def _set_labels(self):
        labels = []
        for point_obj in self._point_obj_arr:
            cluster_number = point_obj.get_cluster_number()
            labels.append(cluster_number)
        labels = np.array(labels)
        self.labels_ = labels

    def get_observation_set(self):
        return self._observation_set

    def get_n_clusters(self):
        return self._n_clusters

    def get_cluster_dict(self):
        return self._cluster_dict

    def get_point_obj_arr(self):
        return self._point_obj_arr

    def get_flattened_cluster_dict(self):
        flattened_dict = {}
        for cluster_index, cluster_obj in self._cluster_dict.items():
            point_arr = cluster_obj.get_point_arr()
            index_arr = [point_obj.get_observation_index(
            ) for point_obj in point_arr]
            index_arr = np.array(index_arr)
            flattened_dict[cluster_index] = index_arr
        return flattened_dict

    def set_n_clusters(self, n_clusters=None):
        self._n_clusters = n_clusters

    def fit(self):
        self._cluster_dict = self._choose_initial_centers()
        cluster_numbers_change = True
        n_iter = 0
        while cluster_numbers_change is True and n_iter <= self._max_iter:
            cluster_numbers_change = False
            for point_obj in self._point_obj_arr:
                cluster_number = point_obj.get_cluster_number()
                cluster_index_to_add = self._compute_closest_cluster_index(point_obj=point_obj,
                                                                           cluster_dict=self._cluster_dict)
                if cluster_number != cluster_index_to_add:
                    self._cluster_dict[cluster_index_to_add].add_point(
                        point_obj=point_obj)
                    if not (cluster_number is None):
                        self._cluster_dict[cluster_number].remove_point(
                            point_obj=point_obj)
                    point_obj.set_cluster_number(
                        cluster_number=cluster_index_to_add)
                    cluster_numbers_change = True
            n_iter += 1
        self._set_inertia()
        self._set_labels()
