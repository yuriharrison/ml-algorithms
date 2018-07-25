"""K Nearest Neighbors Algorithm"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')


def knn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

if __name__ == '__main__':
    dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
    new = np.array([5,7]) 

    result = knn(dataset, new)

    print(result)

    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0],ii[1],s=100,color=i)

    plt.scatter(new[0], new[1],s=100, color=result)
    plt.show()


