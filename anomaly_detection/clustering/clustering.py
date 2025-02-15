import os
import numpy as np
from sklearn import decomposition

import make_graph

class _Algorithm():
    def __init__(self, corpus: list) -> None:
        self.load_corpus(corpus)

    def load_corpus(self, corpus: list) -> None:
        for i in range(len(corpus)):
            corpus[i] = np.load(corpus[i])

        pca = decomposition.PCA(n_components=2)

        # Assume that the first element in the corpus is the training data.
        # Only fit the training data
        self.training_normal = pca.fit_transform(corpus.pop(0))

        self.testing_normal = pca.transform(corpus.pop(0))
        self.testing_attack = pca.transform(corpus.pop(0))

    @staticmethod
    def e_distance(data: np.array) -> np.array:
        '''
        Calculate the euclidean distance between all points in the data set
        '''

        print("Calculating distances")

        n_samples = data.shape[0]
        distances = np.zeros((n_samples, n_samples))

        # Computes the upper triangle of the distance matrix
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                e = np.linalg.norm(data[i] - data[j])
                distances[i, j] = e
                distances[j, i] = e

        np.save('nids_clustering/cache/distances.npy', distances)

        return distances

    def train(self) -> None:
        '''
        Todo: Implement training method
        '''

        raise NotImplementedError()


    def test(self) -> None:
        '''
        Todo: Implement testing method
        '''

        raise NotImplementedError()


    def evalute(self) -> None:
        '''
        Todo: Implement evaluation method
        '''

        raise NotImplementedError()


class K_means(_Algorithm):

    def __init__(self, corpus) -> None:
        super(K_means, self).__init__(corpus)

        self.name = "K Means"


class DBSCAN(_Algorithm):

    def __init__(self, *args, **kwargs) -> None:
        super(DBSCAN, self).__init__(*args, **kwargs)

        self.name = "Density-based spatial clustering of applications with noise"
        self.e = 0.015 # Estimated from elbow plot
        self.min_samples = 4   

        if os.path.exists('nids_clustering/cache/distances.npy'):
            self.e_distance_arr = np.load('nids_clustering/cache/distances.npy')

        else:
            self.e_distance_arr = _Algorithm.e_distance(self.training_normal)

        make_graph.plot_eps(self.e_distance_arr, self.min_samples)

    @staticmethod
    def estimate_eps(data: np.array) -> float:
        '''
        Estimate the optimal value of epsilon for DBSCAN
        '''

        # distances = _Algorithm.e_distance(data)

        # return np.mean(distances)

    def train(self) -> None:
        pass