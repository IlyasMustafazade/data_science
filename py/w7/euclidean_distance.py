import math

import distance


class EuclideanDistance(distance.Distance):
    def get_distance(self, verbose="silent"):
        sum_ = 0
        for i in range(self._len):
            diff = self._object_a[i] - \
                self._object_b[i]
            diff_sq = diff * diff
            sum_ += diff_sq
        dist = math.sqrt(sum_)
        if verbose == "debug":
            print("\nDistance is ", dist, "\n")
        return dist
