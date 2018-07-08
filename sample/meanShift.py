"""MeanShift Algorithm"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')


class MeanShift:

    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
    
    def _define_radius(self, data):
        """Calculate an estimated radius for the given data"""
        all_data_centroid = np.average(data, axis=0)
        all_data_norm = np.linalg.norm(all_data_centroid)
        self.radius = all_data_norm/self.radius_norm_step
        print('Auto defined radius:', self.radius)

    def fit(self, data):
        """Compute Mean Shift clustering"""
        if self.radius == None:
            self._define_radius(data)

        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.00000000001

                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            to_pop = []
            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        #print(np.array(i), np.array(ii))
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

            if optimized:
                break
            
        self.centroids = centroids
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            distances = []
            for centroid in self.centroids:
                norm = np.linalg.norm(featureset - self.centroids[centroid])
                distances.append(norm)

            classification = (distances.index(min(distances)))
            self.classifications[classification].append(featureset)


    def predict(self, data):
        """Predict the closest cluster each sample in data belongs to."""
        distances = []
        for centroid in self.centroids:
            norm = np.linalg.norm(data-self.centroids[centroid])
            distances.append(norm)

        classification = (distances.index(min(distances)))
        return classification


if __name__ == '__main__':
    X, y = make_blobs(n_samples=15, centers=3, n_features=2)

    clf = MeanShift()
    clf.fit(X)

    centroids = clf.centroids
    print(centroids)

    colors = 10*['r','g','b','c','k','y']

    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0],featureset[1], 
                        marker = 'x', 
                        color=color, 
                        s=150, 
                        linewidths = 5, 
                        zorder = 10
                        )

    for c in centroids:
        plt.scatter(centroids[c][0],centroids[c][1], 
                    color='k',
                    marker = '*',
                    s=150,
                    linewidths = 5
                    )

    plt.show()