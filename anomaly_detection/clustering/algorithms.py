import os
import numpy as np
import logging
from sklearn import decomposition, metrics
from scipy.spatial.distance import cdist

logging.basicConfig(
    level=logging.DEBUG,
    format = '%(asctime)s - %(message)s'
)

logger = logging.getLogger(__name__)

class _Algorithm():

    def __init__(self, corpus: list, dimensions: int = 2) -> None:
        self.dimensions = dimensions
        self.load_corpus(corpus)


    def load_corpus(self, corpus: list) -> None:
        '''
        Loads and reduces the dimensionality of the data set using PCA
        '''

        logger.debug("Loading and reducing data")

        pca = decomposition.PCA(n_components=self.dimensions)

        # Reduce the dimensionality of the data
        self.training_normal = pca.fit_transform(np.load(corpus[0]))
        self.testing_normal = pca.transform(np.load(corpus[1]))
        self.testing_attack = pca.transform(np.load(corpus[2]))

        logger.debug("Training data shape: %s", self.training_normal.shape)


    @staticmethod
    def e_distance(data: np.array) -> np.array:
        '''
        Calculate the euclidean distance between all points in the data set
        '''

        logger.debug("Calculating distances")

        distances = metrics.pairwise.euclidean_distances(data)

        np.save('anomaly_detection/cache/distances.npy', distances)

        return distances


    @staticmethod
    def calculate_metrics(TP: int, FP: int, TN: int, FN: int) -> dict:
        '''
        Calculate metrics based on TP, FP, TN, FN
        '''

        try:
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            tp_rate = TP / (TP + FN)
            fp_rate = FP / (TN + FP)
            f1 = (2 * TP) / ((2 * TP) + FP + FN)

        except ZeroDivisionError:
            accuracy = 0
            tp_rate = 0
            fp_rate = 0
            f1 = 0

        return {
            'accuracy': accuracy * 100,
            'tp_rate': tp_rate * 100,
            'fp_rate': fp_rate * 100,
            'f1': f1 * 100
        }


class K_Means(_Algorithm):

    def __init__(self, *args, **kwargs) -> None:
        super(K_Means, self).__init__(*args, **kwargs)

        self.name = "K Means Clustering"
        self.k = 3
        self.tolerance = 1e-4
        self.max_iterations = 100
        self.threshold = np.percentile(self.training_normal, 95)
        self.centroids = self.train()
        self.evalute(self.centroids)


    def train(self) -> np.array:
        '''
        Trains the model
        
        Steps:
        1. Initalizes k centroids at random
        2. Assigns data points to each cluster (nearest centroid)
        3. Loops through computing new centroids for max iterations
        '''

        logger.debug("--- Identifying Centroids ---")

        indices = np.random.choice(self.training_normal.shape[0], self.k, replace=False)

        # Initialize centroids
        centroids = self.training_normal[indices, :]
        iterations = 0

        for _ in range(self.max_iterations):
            iterations += 1 
            
            distances = cdist(self.training_normal, centroids, metric='euclidean')
            assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)

            for cluster in range(self.k):
                cluster_points = self.training_normal[assignments == cluster]
                
                if len(cluster_points) > 0:
                    new_centroids[cluster] = np.mean(cluster_points, axis=0)

                else:
                    distances = cdist(self.training_normal, centroids, metric='euclidean')
                    distances_c = np.min(distances, axis=1)
                    new_centroids[cluster] = self.training_normal[np.argmax(distances_c)]

            if np.all(np.abs(centroids - new_centroids) < self.tolerance):
                logger.debug(f"Converged after {iterations} iterations")
                break

            centroids = new_centroids

        if iterations == self.max_iterations:
            logger.debug(f"Reached maximum iterations: {iterations} without convergance")

        return centroids

    def evalute(self, data):
        '''
        Evaultes the K-Means Model
        
        Steps:
        1. Iterate through each test sample
        2. Compute distance to the nearest centroid
        3. Check distance threshold and see if anomaly
        4. Count metrics 
        5. Print total metrics
        '''

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for sample in self.testing_normal:
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            nearest = np.min(distances)

            if nearest > self.threshold:
                FP += 1
            else:
                TN += 1

        for sample in self.testing_attack:
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            nearest = np.min(distances)

            if nearest > self.threshold:
                TP += 1
            else:
                FN += 1

        metrics = _Algorithm.calculate_metrics(TP, FP, TN, FN)

        logger.debug(f"Accuracy: {metrics['accuracy']:.2f}%")
        logger.debug(f"True Positive Rate: {metrics['tp_rate']:.2f}%")
        logger.debug(f"False Positive Rate: {metrics['fp_rate']:.2f}%")
        logger.debug(f"F1 Score: {metrics['f1']:.2f}%")


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

        metrics = _Algorithm.calculate_metrics(TP, FP, TN, FN)

        logger.debug(f"Accuracy: {metrics['accuracy']:.2f}%")
        logger.debug(f"True Positive Rate: {metrics['tp_rate']:.2f}%")
        logger.debug(f"False Positive Rate: {metrics['fp_rate']:.2f}%")
        logger.debug(f"F1 Score: {metrics['f1']:.2f}%")

        return labels