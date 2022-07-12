import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kmeans


def main():
    df = pd.read_csv("Mall_Customers.csv")
    df = df.drop(
        labels=["Gender", "CustomerID", "Age"], axis=1)
    observation_set = df.values
    n_clusters = 4
    clusterer = kmeans.KMeans(
        observation_set=observation_set, n_clusters=n_clusters, max_iter=300,
        distance_metric="rectilinear")
    clusterer.fit()
    cluster_dict = clusterer._cluster_dict
    new_dict = {i: []
                for i in range(clusterer._n_clusters)}
    for cluster_number, cluster_object in cluster_dict.items():
        observation_arr = cluster_object.get_observation_arr(
            observation_set=observation_set)
        x_vals = []
        y_vals = []
        for observation in observation_arr:
            x_vals.append(observation[0])
            y_vals.append(observation[1])
        plt.scatter(x_vals, y_vals)
    plt.show()


if __name__ == "__main__":
    main()
