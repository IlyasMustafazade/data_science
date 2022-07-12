import distance


class RectilinearDistance(distance.Distance):
    def get_distance(self, verbose="silent"):
        sum_ = 0
        for i in range(self._len):
            diff = self._object_a[i] - \
                self._object_b[i]
            diff_abs = abs(diff)
            sum_ += diff_abs
        dist = sum_
        if verbose == "debug":
            print("\nDistance is ", dist, "\n")
        return dist
