import os
import numpy as np
import random
from sklearn import decomposition
from functools import lru_cache

import graphs

class _Algorithm():

    def __init__(self, corpus) -> None:
        print("Loading and reducing data")
        self.testing_normal_data = _Algorithm.load_corpus(corpus[0])
        self.testing_anomaly_data = _Algorithm.load_corpus(corpus[1])
        self.training_normal_data = _Algorithm.load_corpus(corpus[2])

    @staticmethod
    def load_corpus(path: str) -> np.array:
        data = np.load(path)

        data = _Algorithm.reduce(data)

        return data

    @staticmethod
    def reduce(data: np.array) -> np.array:
        '''
        Reduces the dimensionality of the data set to 2D using PCA
        '''

        print(f"Original shape: {data.shape}")

        pca = decomposition.PCA(n_components=2)

        fit_data = pca.fit_transform(data)

        print(f"Transformed shape: {fit_data.shape}")

        return fit_data

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


class K_Means(_Algorithm):
    def __init__(self, *args, **kwargs) -> None:
        super(K_Means, self).__init__(*args, **kwargs)

        self.name = "K-Means"                                               # Algorithm name
        self.k = 3                                                          # Number of clusters (k)
        self.tolerance = 1e-4                                               # Tolerance for centroid convergence
        self.max_iterations = 100                                           # Maximum iterations for the clustering process
        self.threshold = np.percentile(self.training_normal_reduced, 95)    # Threshold for anomaly detection (95th percentile of normal data)
        self.centroids = self.cluster()                                     # Train the model by identifying cluster centroids
        self.evaluate()                                                     # Evaluate model performance


    def cluster(self) -> np.array:
        '''
        Trains the model
        
        Steps:
        1. Initalizes k centroids at random
        2. Assigns data points to each cluster (nearest centroid)
        3. Loops through computing new centroids for max iterations
        '''
        logger.debug("--- Identifying Centroids ---")

        # Select k random indices to initialize centroids
        indices = np.random.choice(self.training_normal_reduced.shape[0], self.k, replace=False)

        # Initialize centroids with selected data points
        centroids = self.training_normal_reduced[indices, :]
        iterations = 0

        # Run through X iterations to optimize the centroids
        for _ in range(self.max_iterations):
            iterations += 1 
            
            # Compute the Euclidean distance between each data point and centroids
            distances = cdist(self.training_normal_reduced, centroids, metric='euclidean')
            
            # Assign each point to the nearest centroid
            assignments = np.argmin(distances, axis=1)

            # Placeholder for updated centroids
            new_centroids = np.zeros_like(centroids)

            for cluster in range(self.k):
                # Extract points belonging to the current cluster
                cluster_points = self.training_normal_reduced[assignments == cluster]
                
                if len(cluster_points) > 0:
                    # Compute new centroid as the mean of assigned points
                    new_centroids[cluster] = np.mean(cluster_points, axis=0)

                else:
                    # Handle empty cluster: reassign centroid to farthest point
                    distances = cdist(self.training_normal_reduced, centroids, metric='euclidean')
                    distances_c = np.min(distances, axis=1)
                    new_centroids[cluster] = self.training_normal_reduced[np.argmax(distances_c)]

            # Check for convergence (if centroids do not change significantly)
            if np.all(np.abs(centroids - new_centroids) < self.tolerance):
                logger.debug(f"Converged after {iterations} iterations")
                break

            # Update centroids for next iteration
            centroids = new_centroids

        # Log if max iterations are reached without convergence
        if iterations == self.max_iterations:
            logger.debug(f"Reached maximum iterations: {iterations} without convergance")

        return centroids
    
    def calculate_rates(self):
        '''
        Calculates the classification rates: 
        - True Positives (TP): Correctly identified anomalies
        - True Negatives (TN): Correctly identified normal data
        - False Positives (FP): Normal data misclassified as anomalies
        - False Negatives (FN): Anomalies misclassified as normal
        '''
        TP = TN = FP = FN = 0

        # Evaluate normal testing samples
        for sample in self.testing_normal_reduced:
            # Compute distances from centroids
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            
            # Determine the closest centroid
            nearest = np.min(distances)

            # Classify based on threshold (above threshold = anomaly)
            if nearest > self.threshold:
                FP += 1
            else:
                TN += 1

        # Evaluate attack testing samples
        for sample in self.testing_attack_reduced:
            # Compute distances from centroids
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            nearest = np.min(distances)

            # Classify based on threshold (above threshold = anomaly)
            if nearest > self.threshold:
                TP += 1
            else:
                FN += 1

        return TP, TN, FP, FN

class DBSCAN(_Algorithm):

    def __init__(self, *args, **kwargs) -> None:
        super(DBSCAN, self).__init__(*args, **kwargs)

        self.name = "Density-based spatial clustering of applications with noise"
        self.e = 0.0075 # Estimated from elbow plot
        self.min_samples = 4   

        if os.path.exists('nids_clustering/cache/distances.npy'):
            self.e_distance_arr = np.load('nids_clustering/cache/distances.npy')

        else:
            self.e_distance_arr = _Algorithm.e_distance(self.training_normal_data)

        # graphs.plot_eps(self.e_distance_arr, self.min_samples)
        self.clusters = self.train()

        print(f"Clusters identified: {len(self.clusters)}")

        print("--- Clustering normal data ---")
        self.evaluate(self.testing_normal_data)

        print("--- Clustering anomaly data ---")
        self.evaluate(self.testing_anomaly_data)

    def train(self) -> list[list[int]]:
        '''
        Classifies and clusters the core and non-core points of the training data set
        '''

        print("--- Identifying core and non-core points ---")

        core_pts = set()
        non_core_pts = set()

        for i in range(self.training_normal_data.shape[0]):
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
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        n_samples = data.shape[0]
        labels = np.full(n_samples, -1)

        for test_i, test_p in enumerate(data):
            for cluster_i, cluster in enumerate(self.clusters):
                for core_i in cluster:
                    core_p = self.training_normal_data[core_i]

                    distance = np.linalg.norm(test_p - core_p)

                    if distance <= self.e:
                        labels[test_i] = cluster_i
                        break

                if labels[test_i] != -1:
                    break
        
        for i, label in enumerate(labels):
            if i < len(self.testing_anomaly_data):
                if label == -1:
                    TP += 1  
                else:
                    FN += 1  
            else:
                if label == -1:
                    FP += 1 
                else:
                    TN += 1          

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        TPrate = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPrate = FP / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

        print(f"Accuracy: {accuracy:.4f}")
        print(f"True Positive Rate (TPR): {TPrate:.4f}")
        print(f"False Positive Rate (FPR): {FPrate:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

        return labels
