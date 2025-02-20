import os
import numpy as np
import logging
from sklearn import decomposition, metrics
from scipy.spatial.distance import cdist
import utils
from copy import deepcopy

logging.basicConfig(
    level=logging.DEBUG,
    format = '%(asctime)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.propagate = False

class _Algorithm():
    def __init__(self, corpus: list, dimensions: int = 2) -> None:
        self.corpus = deepcopy(corpus) 
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

    def calculate_metrics(self, TP: int, FP: int, TN: int, FN: int) -> dict:
        '''
        Calculate metrics based on TP, FP, TN, FN
        '''

        try:
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            tpr = TP / (TP + FN)
            fpr = FP / (TN + FP)
            f1 = (2 * TP) / ((2 * TP) + FP + FN)

        except ZeroDivisionError:
            accuracy = 0
            tpr = 0
            fpr = 0
            f1 = 0

        self.metrics = {
            'accuracy': accuracy * 100,
            'tpr': tpr * 100,
            'fpr': fpr * 100,
            'f1': f1 * 100
        }

    def evaluate(self):
        TP, TN, FP, FN = self.calculate_rates()

        self.calculate_metrics(TP, FP, TN, FN)

        print(self.name)
        print(f"Accuracy: {self.metrics['accuracy']:.2f}%")
        print(f"True Positive Rate: {self.metrics['tpr']:.2f}%")
        print(f"False Positive Rate: {self.metrics['fpr']:.2f}%")
        print(f"F1 Score: {self.metrics['f1']:.2f}%")
        print()

    def cluster(self):
        raise NotImplementedError
    
    def calculate_rates(self):
        raise NotImplementedError

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

        self.name = "DBScan"
        self.e = 0.0075 # Estimated from elbow plot
        self.min_samples = 4   

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

        logger.debug(f"Clusters identified: {len(self.clusters)}")

        # I don't think we need to separate evaluations
        '''
        print("--- Clustering normal data ---")
        self.evaluate(self.testing_normal_reduced)
        '''
        
        self.evaluate()

    def cluster(self) -> list[list[int]]:
        '''
        Classifies and clusters the core and non-core points of the training data set
        '''

        logger.debug("--- Identifying core and non-core points ---")

        core_pts = set()
        non_core_pts = set()

        for i in range(self.training_normal_reduced.shape[0]):
            distance = self.distances[i]

            n = np.sum(distance < self.e)

            if n <= self.min_samples:
                non_core_pts.add(i)
            else: 
                core_pts.add(i)

        logger.debug(f"Core points: {len(core_pts)}")
        logger.debug(f"Non-core points: {len(non_core_pts)}")

        logger.debug(" -- Clustering core points --")

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
    
    def calculate_rates(self):
        TP = TN = FP = FN = 0

        # Evaluate each dataset separately
        for data, is_attack in [(self.testing_attack_reduced, True), (self.testing_normal_reduced, False)]:
            n_samples = data.shape[0]
            labels = np.full(n_samples, -1)

            for test_i, test_p in enumerate(data):
                for cluster_i, cluster in enumerate(self.clusters):
                    for core_i in cluster:
                        core_p = self.training_normal_reduced[core_i]
                        distance = np.linalg.norm(test_p - core_p)

                        if distance <= self.e:
                            labels[test_i] = cluster_i
                            break

                    if labels[test_i] != -1:
                        break

            # Evaluate based on the type of data
            for _, label in enumerate(labels):
                if is_attack:
                    if label == -1:
                        TP += 1  
                    else:
                        FN += 1
                else:
                    if label == -1:
                        FP += 1
                    else:
                        TN += 1

        return TP, TN, FP, FN