import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from processing import Data

class Cluster:
    def __init__(self, data: Data, model_name: str, kwargs):
        self.data = data
        self.kwargs = kwargs
        self.model_name = model_name

        self.clusters = None
        self.n_clusters = 0

        self.silhouette = 0
        self.calinski = 0

    def fit(self, **kwargs):
        print(f'Clustering using {self.model_name} model...')

        if self.model_name == 'kmeans':
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
        '''
            Draw graph which visualizes the results of the model
            Labels each cluster by the majority of traffic in it
        '''
        
        filepath = filepath + f'/{self.model_name}/'
        os.makedirs(filepath, exist_ok=True)
        filepath = filepath + f'{str(self.n_clusters)}.png'

        # Reduce the number of components to 2 so it is 2D
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(self.features)
        
        # Plot each cluster
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 8), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')

        unique_labels = np.unique(self.clusters)
        used_labels = []

        i = -1
        for label in unique_labels:
            i += 1
            mask = (self.clusters == label)

            if label == -1:
                label_name = 'Noise'

            elif self.model_name == 'kmeans':
                codes = self.labels[mask] 
                if codes.size > 0:
                    majority_code = np.bincount(codes).argmax()
                    # map code back to family name based on most prevalent traffic in each cluster
                    label_name = f'Majority {self.data.le.inverse_transform([majority_code])[0]}'
                else:
                    label_name = 'Empty'
            else:
                label_name = f'Cluster {label}'
                
            if label_name in used_labels:
                label_name = f'{label_name} (Cluster {used_labels.count(label_name)})'
            used_labels.append(label_name)
            
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=30,
                alpha=0.6,
                label=label_name,
                color=plt.cm.Set3(i % 12),
            )
            
        plt.title(f'{self.model_name.capitalize()} Cluster Plot ({self.n_clusters} clusters)', 
                color='white', pad=20)
        plt.xlabel('Principal Component 1', color='white')
        plt.ylabel('Principal Component 2', color='white')
        plt.tick_params(colors='white')
        
        legend = plt.legend(facecolor='black', labelcolor='white')
        plt.setp(legend.get_frame(), color='black')
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor='black', bbox_inches='tight')
        plt.close()
        
        plt.style.use('default')