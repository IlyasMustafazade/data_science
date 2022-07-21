import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy


class Dendrogram():
    @staticmethod
    def plot_dendrogram(model=None, **kwargs):
        plt.figure(num="Hierarchical dendrogram")
        counts = np.zeros(
            model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
        hierarchy.dendrogram(
            linkage_matrix, **kwargs)
