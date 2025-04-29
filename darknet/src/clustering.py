import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from processing import Data

class Cluster:
    def __init__(self, data: Data, model_name: str):
        self.data = data
        self.features = data.X_train_scaled 

        self.model_name = model_name

        self.clusters = []
        self.n_clusters = 0

        self.silhouette = 0
        self.calinski = 0

    def fit(self, **kwargs):
        if self.model_name == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 8)
            self.model = KMeans(n_clusters=n_clusters)
        
        elif (self.model_name == 'dbscan'):
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)

            self.model = DBSCAN(eps=eps, min_samples=min_samples)

        self.clusters = self.model.fit_predict(self.features)

    def evaluate(self):
        self.n_clusters = len(np.unique(self.clusters))

        if self.n_clusters > 1:
            self.silhouette = silhouette_score(self.features, self.clusters)
            self.calinski = calinski_harabasz_score(self.features, self.clusters)
        else:
            self.silhouette = 0
            self.calinski = 0

    def draw(self):
        # TODO - draw cluster plots
        pass