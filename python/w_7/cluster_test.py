import numpy as np

import cluster


def main():
    point_arr = np.array(
        [np.array([1, 2]), np.array([2, 3]), np.array([2, 5])])
    cluster_obj = cluster.Cluster(
        point_arr=point_arr)
    cluster_obj.remove_point(
        point=np.array([2, 3]))


if __name__ == "__main__":
    main()
