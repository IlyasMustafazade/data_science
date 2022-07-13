import numpy as np

import distance_chooser
import unsupervised_metrics


class Silhouette(unsupervised_metrics.UnsupervisedMetrics):
    def __init__(self, point_arr=None, cluster_dict=None, distance_metric=None):
        self._point_arr = point_arr
        self._cluster_dict = cluster_dict
        self._distance_metric = distance_chooser.DistanceChooser.choose_distance_metric(
            distance_metric_as_str=distance_metric)

    def _comp_a_i(self, point_i=None, cluster_number=None, cluster_obj=None):
        point_arr = cluster_obj.get_point_arr()
        n_points = cluster_obj.get_n_points()
        divider = n_points - 1
        sigma = 0
        for point_obj in point_arr:
            dist_obj = self._distance_metric(object_a=point_obj.get_observation(),
                                             object_b=point_i.get_observation())
            dist = dist_obj.get_distance()
            sigma += dist
        a_i = sigma / divider
        return a_i

    def _get_between_cluster_distance(self, point_i=None, cluster_obj=None):
        point_arr = cluster_obj.get_point_arr()
        n_points = cluster_obj.get_n_points()
        sigma = 0
        for point_obj in point_arr:
            dist_obj = self._distance_metric(object_a=point_obj.get_observation(),
                                             object_b=point_i.get_observation())
            dist = dist_obj.get_distance()
            sigma += dist
        mean_distance = sigma / n_points
        return mean_distance

    def _comp_b_i(self, point_i=None, cluster_number=None, cluster_i=None):
        between_cluster_distance_arr = []
        for cluster_obj in self._cluster_dict.values():
            if not(cluster_obj is cluster_i):
                between_cluster_distance = self._get_between_cluster_distance(
                    point_i=point_i, cluster_obj=cluster_obj)
                between_cluster_distance_arr.append(
                    between_cluster_distance)
        if len(between_cluster_distance_arr) > 0:
            min_between_cluster_distance = min(
                between_cluster_distance_arr)
        else:
            min_between_cluster_distance = 0
        return min_between_cluster_distance

    def silhouette_observation(self, point_obj=None):
        cluster_number = point_obj.get_cluster_number()
        cluster_obj = self._cluster_dict[cluster_number]
        n_points = cluster_obj.get_n_points()
        s_i = 0
        if n_points == 1:
            return s_i
        a_i = self._comp_a_i(
            point_i=point_obj, cluster_number=cluster_number, cluster_obj=cluster_obj)
        b_i = self._comp_b_i(
            point_i=point_obj, cluster_number=cluster_number, cluster_i=cluster_obj)
        if a_i < b_i:
            s_i = 1 - a_i / b_i
        elif a_i == b_i:
            s_i = 0
        elif a_i > b_i:
            s_i = b_i / a_i - 1
        return s_i

    def silhouette_samples(self):
        silhouette_arr = []
        for point_obj in self._point_arr:
            silhouette_coeff = self.silhouette_observation(
                point_obj=point_obj)
            silhouette_arr.append(
                silhouette_coeff)
        silhouette_arr = np.array(silhouette_arr)
        return silhouette_arr

    def silhouette_score(self):
        silhouette_arr = self.silhouette_samples()
        mean_score = np.mean(silhouette_arr)
        return mean_score
