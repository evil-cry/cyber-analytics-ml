import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

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
        """
            Draw graph which visualizes the results of the model
        """
        
        # Reduce the number of components to 2 so it is 2D
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(self.features)
        
        # Plot each cluster
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(self.clusters)
        for label in unique_labels:
            mask = (self.clusters == label)
            if label == -1:
                label_name = 'Noise'
            else:
                label_name = f'Cluster {label}'
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=30,
                alpha=0.6,
                label=label_name
            )
            
        # Make it nice looking
        plt.title(f'{self.model_name.capitalize()} Cluster Plot ({self.n_clusters} clusters)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.tight_layout()
        plt.show()