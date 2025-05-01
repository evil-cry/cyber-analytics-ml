import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from processing import Data

class Cluster:
    def __init__(self, data: Data, model_name: str, **kwargs):
        self.data = data

        self.X_train, self.Y_train, self.X_test, self.Y_test = data.set_get_X_Y(
            kwargs.get('what_to_classify', 'class'),
        )

        self.features = np.vstack((data.X_train_scaled, data.X_test_scaled))
        
        self.labels = np.concatenate((data.Y_train, data.Y_test))

        self.model_name = model_name

        self.clusters = None
        self.n_clusters = 0

        self.silhouette = 0
        self.calinski = 0

    def fit(self, **kwargs):
        if self.model_name == 'kmeans':
            print("Training kmeans model...")
            n_clusters = kwargs.get('n_clusters', 8)
            self.model = KMeans(n_clusters=n_clusters)
            self.clusters = self.model.fit_predict(self.features)
        
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

    def draw(self, filepath='darknet/graphs/clusters'):
        """
            Draw graph which visualizes the results of the model
            Labels each cluster by the majority of traffic in it
        """
        
        filepath = filepath + f'/{self.model_name}/'
        os.makedirs(filepath, exist_ok=True)
        filepath = filepath + f'{str(self.n_clusters)}.png'

        # Reduce the number of components to 2 so it is 2D
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(self.features)
        
        # Plot each cluster
        plt.figure(figsize=(8, 8))
        unique_labels = np.unique(self.clusters)
        for label in unique_labels:
            mask = (self.clusters == label)
            if label == -1:
                label_name = 'Noise'
            elif self.model_name == 'kmeans':
                # find majority of traffic in cluster
                codes = self.labels[mask] 
                if codes.size > 0:
                    majority_code = np.bincount(codes).argmax()
                    # map code back to family name based on most prevalent traffic in each cluster
                    label_name = self.data.le.inverse_transform([majority_code])[0]
                else:
                    label_name = "Empty"
                '''
                family_codes = self.data.label_family[mask]
                if family_codes.size > 0:
                    majority_family = np.bincount(family_codes).argmax()
                    label_name = self.data.family_le.inverse_transform([majority_family])[0]
                else:
                    label_name = "Empty"
                '''
            else:
                label_name = f"Cluster {label}"
                
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=30,
                alpha=0.6,
                label=label_name
            )
            
        # Make it nice looking
        plt.title(f'{self.model_name.capitalize()} Cluster Plot ({self.n_clusters} clusters)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()