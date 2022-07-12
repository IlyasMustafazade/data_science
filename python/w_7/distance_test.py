import numpy as np
import pandas as pd

import distance
import euclidean_distance
import rectilinear_distance


def main():
    object_a = np.array([1, 7.8])
    object_b = np.array([2, 4])
    dist = rectilinear_distance.RectilinearDistance(
        object_a=object_a, object_b=object_b)
    dist.get_objects(verbose="debug")
    dist.get_len(verbose="debug")
    dist.get_distance(verbose="debug")


if __name__ == "__main__":
    main()
