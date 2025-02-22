# AI Usage Statement
# Tools Used: o1, o3-mini, o1-mini, gpt-4o (no one could answer my question)
# - Usage: Transforming data into 2 dimensions before plotting
# - Verification: Comparing the plots before and after the change
# Prohibited Use Compliance: Confirmed

import numpy as np
import logging
from sklearn import decomposition, metrics
from scipy.spatial.distance import cdist
import utils
from copy import deepcopy
import make_graph

logging.basicConfig(
    level=logging.DEBUG,
    format = '%(asctime)s - %(message)s'
)

logger = logging.getLogger(__name__)
logging.disable(50)

class _Algorithm():
    def __init__(self, corpus: list, dimensions: int = 18, plot_dimensions:int = 2, parameters = {}) -> None:
        self.corpus = deepcopy(corpus) 
        self.dimensions = dimensions
        self.parameters = parameters

        self.plot = make_graph.Plot(plot_dimensions)
        self.load_corpus()

    def load_corpus(self) -> None:
        for i in range(len(self.corpus)):
            self.corpus[i] = np.load(self.corpus[i])

        # Assume that the first element in the corpus is the training data.
        self.training_normal = self.corpus[0]
        self.testing_normal = self.corpus[1]
        self.testing_attack = self.corpus[2]

        pca = decomposition.PCA(n_components=self.dimensions)
        plot_pca = decomposition.PCA(n_components=self.plot.dimensions)

        # Only fit the training data
        pca.fit(self.training_normal)
        self.pca = pca

        self.training_normal_reduced = pca.transform(self.training_normal)
        self.testing_normal_reduced = pca.transform(self.testing_normal)
        self.testing_attack_reduced = pca.transform(self.testing_attack)
            
        plot_pca.fit(self.training_normal_reduced)
        self.plot_pca = plot_pca

    def calculate_metrics(self) -> dict:
        '''
        Calculate metrics based on TP, FP, TN, FN
        '''
        TP, TN, FP, FN = self.TP, self.TN, self.FP, self.FN
        try:
            self.accuracy = (TP + TN) / (TP + FP + TN + FN)
            self.tpr = TP / (TP + FN)
            self.fpr = FP / (TN + FP)
            self.f1 = (2 * TP) / ((2 * TP) + FP + FN)

        except ZeroDivisionError:
            self.accuracy = 0
            self.tpr = 0
            self.fpr = 0
            self.f1 = 0

        self.metrics = {
            'accuracy': self.accuracy * 100,
            'tpr': self.tpr * 100,
            'fpr': self.fpr * 100,
            'f1': self.f1 * 100
        }

    def evaluate(self):
        self.calculate_rates()
        self.calculate_metrics()

        print(self.name)
        print(f"Clustering results: TP={self.TP}, FP={self.FP}, TN={self.TN}, FN={self.FN}")
        print(f"Accuracy: {self.metrics['accuracy']:.2f}%")
        print(f"True Positive Rate: {self.metrics['tpr']:.2f}%")
        print(f"False Positive Rate: {self.metrics['fpr']:.2f}%")
        print(f"F1 Score: {self.metrics['f1']:.2f}%")
        print()

    def cluster(self):
        raise NotImplementedError
    
    def calculate_rates(self):
        raise NotImplementedError
    
    def draw(self, path):
        self.plot.draw(path)

    def print_score(self, score):
        params_str = ",".join(f"{k}={v:.4f}" for k, v in self.parameters.items())
        return f'{self.name}:{params_str}:{score}\n'

class K_Means(_Algorithm):
    '''
    K-Means Clustering Algorithm
    Optional parameters:
        k: int, default = 79
           Number of clusters (k)
        tolerance: float, default = 0.0001
            Tolerance for centroid convergence
        max: int, default = 100
           Maximum iterations for the clustering process
        threshold: float, default = 95
            Threshold for anomaly detection (95th percentile of normal data)
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(K_Means, self).__init__(*args, **kwargs)
        self.name = "K-Means"
        self.plot.configure('X', 'Y', title=f"{self.name}:{self.parameters}")
        self.k = self.parameters.get('k') or 79
        self.tolerance = self.parameters.get('tolerance') or 0.0001
        self.max_iterations = self.parameters.get('max') or 100

        threshold = self.parameters.get('threshold') or 95
        self.threshold = np.percentile(self.training_normal_reduced, threshold)

        self.centroids = self.cluster()
        self.evaluate()


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
        self.TP = self.TN = self.FP = self.FN = 0

        # Evaluate normal testing samples
        for sample in self.testing_normal_reduced:


            '''
            This single line of code made me go into an hour-long research on how to get the data into two dimension before plotting.
            At first, I didn't think of simply transforming a single sample. I couldn't find anything on the internet regarding this - our use case seems pretty rare.
            I don't use AIs for coding this, only for general questions regarding algorithms.
            Not a single model I tried answered my question in a helpful way. 
            Finally, through experimentation and lots of wasted openai tokens (a whopping 50 cents worth of them), this line was made. 
            '''
            sample_2d = self.plot_pca.transform(sample.reshape(1, -1))[0]


            # Compute distances from centroids
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            
            # Determine the closest centroid
            nearest = np.min(distances)

            # Classify based on threshold (above threshold = anomaly)
            if nearest > self.threshold:
                self.FP += 1
                self.plot += (sample_2d, 'fp')
            else:
                self.TN += 1
                self.plot += (sample_2d, 'tn')

        # Evaluate attack testing samples
        for sample in self.testing_attack_reduced:

            sample_2d = self.plot_pca.transform(sample.reshape(1, -1))[0]
            # Compute distances from centroids
            distances = np.linalg.norm(sample - self.centroids, axis=1)
            nearest = np.min(distances)

            # Classify based on threshold (above threshold = anomaly)
            if nearest > self.threshold:
                self.TP += 1
                self.plot += (sample_2d, 'tp')
            else:
                self.FN += 1
                self.plot += (sample_2d, 'fn')

class DBSCAN(_Algorithm):
    '''
    DBScan Clustering Algorithm
    Optional parameters:
        e: float, default = 0.0146
        Epsilon
        min: int, default = 2
            minimum samples required to form a cluster
            https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd - This suggest that high min should work best, for some reason that's not the case
        p: utils.Distance, default = euclidean distance
        distance function to use. Check @utils.py for info
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(DBSCAN, self).__init__(*args, **kwargs)

        self.name = "DBScan"
        self.plot.configure('X', 'Y', title=f"{self.name}:{self.parameters}")
        self.e = self.parameters.get('e') or 0.0146 # Estimated from elbow plot
        self.min_samples = self.parameters.get('min') or 2

        if 'p' in self.parameters:
            p = self.parameters.get('p')
            self.distances = utils.Distance(p)
        else:
            self.distances = utils.Distance(False)

        if not self.distances:
            # Only pass distance if it is given
            self.distances.calculate(self.training_normal_reduced, **({'d': self.parameters['d']} if 'd' in self.parameters else {}))

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
        '''
        I found the error in the algorithm and fixed it
        You were checking if the point didn't belong to any of the clusters, which resulted in like 0.1% of anomalies
        I also optimized it by checking all the distances at once
        You can delete this when making the doc
        '''
        self.TP = self.TN = self.FP = self.FN = 0

        # if you could think of different variable names it would be great!

        # Evaluate each dataset separately and sum the values
        for data, is_attack in [(self.testing_attack_reduced, True), (self.testing_normal_reduced, False)]:
            for test_point in data:

                # Count all neighbors within epsilon radius
                distances = cdist([test_point], self.training_normal_reduced)[0]
                # Count how many points are within epsilon distance of the test point
                neighbors = np.sum(distances <= self.e)
                
                # Point is an anomaly if it has fewer neighbors than min_samples
                is_anomaly = neighbors < self.min_samples
                
                sample_2d = self.plot_pca.transform(test_point.reshape(1, -1))[0]
                
                if is_attack:
                    if is_anomaly:
                        self.TP += 1
                        self.plot += (sample_2d, 'tp')
                    else:
                        self.FN += 1
                        self.plot += (sample_2d, 'fn')
                else:
                    if is_anomaly:
                        self.FP += 1
                        self.plot += (sample_2d, 'fp')
                    else:
                        self.TN += 1
                        self.plot += (sample_2d, 'tn')