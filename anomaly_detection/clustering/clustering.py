import os
import numpy as np
from sklearn import decomposition, metrics

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

        distances = metrics.pairwise.euclidean_distances(data)

        np.save('anomaly_detection/cache/distances.npy', distances)

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
        self.e = 0.0075 # Estimated from elbow plot
        self.min_samples = 4   

        if os.path.exists('anomaly_detecion/cache/distances.npy'):
            self.e_distance_arr = np.load('anomaly_detecion/cache/distances.npy')

        else:
            self.e_distance_arr = _Algorithm.e_distance(self.training_normal)

        # graphs.plot_eps(self.e_distance_arr, self.min_samples)
        self.clusters = self.train()

        print(f"Clusters identified: {len(self.clusters)}")

        print("--- Clustering normal data ---")
        self.evaluate(self.testing_normal)

        print("--- Clustering anomaly data ---")
        self.evaluate(self.testing_attack)

    @staticmethod
    def estimate_eps(data: np.array) -> float:
        '''
        Estimate the optimal value of epsilon for DBSCAN
        '''

        raise NotImplementedError()    

    def train(self) -> list[list[int]]:
        '''
        Classifies and clusters the core and non-core points of the training data set
        '''

        print("--- Identifying core and non-core points ---")

        core_pts = set()
        non_core_pts = set()

        for i in range(self.training_normal.shape[0]):
            distance = self.e_distance_arr[i]

            n = np.sum(distance < self.e)

            if n <= self.min_samples:
                non_core_pts.add(i)
            else: 
                core_pts.add(i)

        print(f"Core points: {len(core_pts)}")
        print(f"Non-core points: {len(non_core_pts)}")

        print(" -- Clustering core points --")

        visited = set()
        clusters = []

        # Cluster core points
        for core in core_pts:
            if core not in visited:
                # Create new cluster
                cluster = []
                unchecked_points = [core]

                while unchecked_points:
                    p = unchecked_points.pop()

                    if p not in visited:
                        visited.add(p)
                        cluster.append(p)

                        # Get the neighbors of the current point
                        distance = self.e_distance_arr[p]
                        n = np.where(distance < self.e)[0]

                        # Determine if the point is a core point
                        if p in core_pts:
                            for neighbor in n:
                                if neighbor not in visited:
                                    unchecked_points.append(int(neighbor))

                # Add cluster to list of clusters
                clusters.append(cluster)

        return clusters
    
    def evaluate(self, data) -> list[list[int]]:
        '''
        Classify the testing data set
        '''

        n_samples = data.shape[0]
        labels = np.full(n_samples, -1)

        for test_i, test_p in enumerate(data):
            for cluster_i, cluster in enumerate(self.clusters):
                for core_i in cluster:
                    core_p = self.training_normal[core_i]

                    distance = np.linalg.norm(test_p - core_p)

                    if distance <= self.e:
                        labels[test_i] = cluster_i
                        break

                if labels[test_i] != -1:
                    break
                    
        anomalies = []

        for p in labels:
            if p == -1:
                anomalies.append(p)

        print(f"Anomalies detected: {len(anomalies)}")

        return labels