import euclidean_distance
import rectilinear_distance


class DistanceChooser():
    @staticmethod
    def choose_distance_metric(distance_metric_as_str=None):
        if distance_metric_as_str == "euclidean":
            return euclidean_distance.EuclideanDistance
        elif distance_metric_as_str == "rectilinear":
            return rectilinear_distance.RectilinearDistance
        else:
            raise Exception(
                "Unknown distance metric")
