import matplotlib.pyplot as plt
from sklearn import metrics


class ClusterReporter():
    @staticmethod
    def report(n_clusters=None, clusterer=None,
               clusterer_name=None, observation_set=None, column=None):
        cluster_arr = [[]
                       for i in range(n_clusters)]
        label_index = 0
        clusterer_labels = clusterer.labels_
        for label in clusterer.labels_:
            cluster_arr[label].append(label_index)
            label_index += 1
        for i in range(n_clusters):
            print("\nCluster", str(i), "-> ",
                  column[cluster_arr[i]])
        plt.figure(num=clusterer_name)
        for cluster_ in cluster_arr:
            x_vals = []
            y_vals = []
            for index in cluster_:
                x_vals.append(
                    observation_set[index][0])
                y_vals.append(
                    observation_set[index][1])
            plt.scatter(x_vals, y_vals)
        print("\n", clusterer_name, "'s silhouette score -> ",
              metrics.silhouette_score(observation_set, clusterer_labels), "\n")
        print("\n", clusterer_name, "'s CH score -> ",
              metrics.calinski_harabasz_score(observation_set, clusterer_labels), "\n")
