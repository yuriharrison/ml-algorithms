"""K-Means clustering Algorithm"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class KMeans:
    """K-Means clustering

    -- Arguments
        k: int, optional, default 2
            The number of clusters to form as well as the number of centroids to generate.
        max_iter: int, optional, default 300
            Maximum number of iterations of the k-means algorithm to run.
        tol: float, optional
            The relative increment in the results before declaring convergence.
    """

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        """Compute k-means clustering."""
        self.centroids = centroids = dict()

        for i in range(self.k):
            centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = classifications = dict()

            for i in range(self.k):
                classifications[i] = []

            for featureset in data:
                distances = []
                for centroid in centroids:
                    n = np.linalg.norm(featureset - centroids[centroid])
                    distances.append(n)

                classification = distances.index(min(distances))
                classifications[classification].append(featureset)

            prev_centroids = dict(centroids)

            for classification in classifications:
                centroids[classification] = np.average(classifications[classification],
                                                        axis=0)

            optimized = True

            for c in centroids:
                original_centroid = prev_centroids[c]
                current_centroid = centroids[c]
                s = np.sum((current_centroid - original_centroid)
                            / original_centroid*100)
                if s > self.tol:
                    print(s)
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        """Predict the closest cluster each sample in data belongs to."""
        distances = []
        for centroid in self.centroids:
            n = np.linalg.norm(data-self.centroids[centroid])
            distances.append(n)

        classification = distances.index(min(distances))
        return classification


if '__main__' == __name__:
    main_data = np.array([[1, 2],
             [1.5, 1.8],
             [5, 8],
             [8, 8],
             [1, 0.6],
             [9, 11]])

    colors = ['r','g','b','c','k','o','y']

    clf = KMeans()
    clf.fit(main_data)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=150, linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]

        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", 
                        color=color, s=150, linewidths=5)

    new_data = np.array([[1,3],
                        [8,9],
                        [0,3],
                        [5,4],
                        [6,4]])

    for item in new_data:
        classification = clf.predict(item)
        plt.scatter(item[0], item[1], marker="*", color=colors[classification], 
                    s=150, linewidths=5)

    plt.show()