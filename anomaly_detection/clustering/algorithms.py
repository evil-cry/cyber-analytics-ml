import os
import numpy as np
from sklearn import decomposition, metrics
import utils

class _Algorithm():
    def __init__(self, corpus: list, dimensions: int = 2) -> None:
        self.corpus = corpus
        self.dimensions = dimensions
        self.load_corpus()

        self.normal_count = len(self.testing_normal_reduced)
        self.attack_count = len(self.testing_attack_reduced)

    def load_corpus(self) -> None:
        for i in range(len(self.corpus)):
            self.corpus[i] = np.load(self.corpus[i])

        # Assume that the first element in the corpus is the training data.
        self.training_normal = self.corpus[0]
        self.testing_normal = self.corpus[1]
        self.testing_attack = self.corpus[2]

        pca = decomposition.PCA(n_components=self.dimensions)

        # Only fit the training data
        self.training_normal_reduced = pca.fit_transform(self.training_normal)
        self.testing_normal_reduced = pca.transform(self.testing_normal)
        self.testing_attack_reduced = pca.transform(self.testing_attack)

    def cluster(self) -> None:
        raise NotImplementedError()

    def evalute(self) -> None:
        raise NotImplementedError()

class K_means(_Algorithm):
    def __init__(self, corpus) -> None:
        super(K_means, self).__init__(corpus)

        self.name = "K-Means"


# https://www.dbs.ifi.lmu.de/Publikationen/Papers/VLDB-98-IncDBSCAN.pdf
class DBSCAN(_Algorithm):
    def __init__(self, *args, **kwargs) -> None:
        super(DBSCAN, self).__init__(*args, **kwargs)

        self.name = "DBScan"
        self.e = 0.0075 # Estimated from elbow plot
        self.min_samples = 4   
        self.distances = []

        if 'p' in kwargs:
            p = kwargs.get('p')
            self.distances = utils.Distance(p)
        else:
            self.distances = utils.Distance(False)

        if not self.distances:
            # Only pass distance if it is given
            self.distances.calculate(self.training_normal_reduced, **({'d': kwargs['d']} if 'd' in kwargs else {}))

        # graphs.plot_eps(self.e_distance_arr, self.min_samples)
        self.clusters = self.cluster()

        print(f"Clusters: {len(self.clusters)}")

        print("--- Clustering normal data ---")
        self.evaluate(self.testing_normal)

        print("--- Clustering anomaly data ---")
        self.evaluate(self.testing_attack)

    def cluster(self) -> None:
        '''
        Classifies and clusters the core and non-core points of the training data set
        '''

        core_pts = set()
        non_core_pts = set()
        clusters = []

        for i in range(self.training_normal.shape[0]):
            distance = self.distances[i]

            n = np.sum(distance < self.e)

            if n <= self.min_samples:
                non_core_pts.add(i)
            else: 
                core_pts.add(i)

        print(f"Core points: {len(core_pts)}")
        print(f"Non-Core points: {len(non_core_pts)}")

        visited = set()
        self.clusters = []

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
                        distance = self.distances[p]
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
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            
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
            
            for i, label in enumerate(labels):
                if i < len(self.testing_attack):
                    if label == -1:
                        TP += 1  
                    else:
                        FN += 1  
                else:
                    if label == -1:
                        FP += 1 
                    else:
                        TN += 1          

            accuracy = (TP + TN) * 100 / (TP + FP + TN + FN)
            tprate = TP * 100 / (TP + FN) if (TP + FN) > 0 else 0
            fprate = FP * 100 / (TN + FP) if (TN + FP) > 0 else 0
            f1 = (2 * TP) * 100 / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

            print(f"Accuracy: {accuracy}%")
            print(f"True Positive Rate (TPR): {tprate}%")
            print(f"False Positive Rate (FPR): {fprate}%")
            print(f"F1 Score: {f1}%")

            return labels
