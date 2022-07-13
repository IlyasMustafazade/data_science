import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as skl_cluster
from sklearn import decomposition

import elbow
import kmeans
import point_array_extractor
import silhouette
from funcs import *


def main():
    file_name = "Country-data"
    file_ext = ".csv"
    full_name = file_name + file_ext
    df = pd.read_csv(full_name)
    country_arr = np.array(df["country"])
    df = df.drop(labels="country", axis=1)
    df_columns = df.columns
    to_normalize = df_columns[1:]
    df = normalize(
        f_space=df, col_arr=to_normalize)
    observation_set = df.values
    point_obj_arr = point_array_extractor.PointArrayExtractor.extract_point_arr(
        observation_set=observation_set)
    # KMEANS CLUSTERING WITH PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(observation_set)
    observation_set = pca.transform(
        observation_set)
    distance_metric = "euclidean"
    n_clusters = 3
    my_kmeans_clusterer = kmeans.KMeans(
        observation_set=observation_set, max_iter=300, distance_metric=distance_metric,
        n_clusters=n_clusters)
    kmeans_clusterer = skl_cluster.KMeans(
        n_clusters=n_clusters)
    my_kmeans_clusterer.fit()
    kmeans_clusterer.fit(observation_set)
    flattened_cluster_dict = my_kmeans_clusterer.get_flattened_cluster_dict()
    cluster_dict = my_kmeans_clusterer.get_cluster_dict()
    for i in range(n_clusters):
        print("\nCluster", str(i), "-> ",
              country_arr[flattened_cluster_dict[i]])
    kmeans_clusterer_labels_ = kmeans_clusterer.labels_
    silhouetter = silhouette.Silhouette(
        point_arr=my_kmeans_clusterer.get_point_obj_arr(),
        cluster_dict=cluster_dict, distance_metric=distance_metric)
    s_score = silhouetter.silhouette_score()
    print("\nSilhouette score -> ",
          s_score, "\n")
    print("\nCH score -> ",
          metrics.calinski_harabasz_score(observation_set, my_kmeans_clusterer.labels_), "\n")
    clustering_results(n_clusters=n_clusters, clusterer=kmeans_clusterer,
                       clusterer_name="Sklearn KMeans", observation_set=observation_set, column=country_arr)
    plt.figure(num="Kmeans clustering")
    for cluster_number, cluster_object in cluster_dict.items():
        observation_arr = cluster_object.get_observation_arr(
            observation_set=observation_set)
        x_vals = []
        y_vals = []
        for observation in observation_arr:
            x_vals.append(observation[0])
            y_vals.append(observation[1])
        plt.scatter(x_vals, y_vals)
    # HIERARCHICAL CLUSTERING WITH PCA
    hierarchical_clusterer = skl_cluster.AgglomerativeClustering(
        n_clusters=n_clusters)
    hierarchical_clusterer.fit(observation_set)
    clustering_results(n_clusters=n_clusters, clusterer=hierarchical_clusterer,
                       clusterer_name="Hierarchical", observation_set=observation_set, column=country_arr)
    plt.show()


if __name__ == "__main__":
    main()
