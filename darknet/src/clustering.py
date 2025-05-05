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

        what_to_classify = kwargs.get('what_to_classify', 'class')
        self.X_train, self.X_test, self.Y_train, self.Y_test = data.set_get_X_Y(
            what_to_classify=what_to_classify,
            max_samples=kwargs.get('max_samples', -1)
        )

        self.features = np.vstack((data.X_train_scaled, data.X_test_scaled))
        self.labels = np.concatenate((self.Y_train, self.Y_test))

        self.clusters = None
        self.n_clusters = 0

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

    def draw(self, filepath='darknet/graphs/clusters', darkmode = True):
        '''
            Draw graph which visualizes the results of the model
            Labels each cluster by the majority of traffic in it
        '''
        
        filepath = filepath + f'/{self.model_name}/'
        os.makedirs(filepath, exist_ok=True)
        filepath = filepath + f'{str(self.n_clusters)}' + f'{str(self.kwargs.get("max_features", ""))}.png'

        # Reduce the number of components to 2 so it is 2D
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(self.features)
        
        # Plot each cluster
        plt.style.use('dark_background') if darkmode else plt.style.use('default')
        plt.figure(figsize=(8, 8), facecolor='black' if darkmode else 'white')
        ax = plt.gca()
        ax.set_facecolor('black' if darkmode else 'white')

        unique_labels = np.unique(self.clusters)
        used_labels = []

        for i, label in enumerate(unique_labels):
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
                label_name = f'{label_name} (Cluster {i})'
            used_labels.append(label_name)
            
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                s=30,
                alpha=0.6,
                label=label_name,
                color=plt.cm.Set3(i % 12) if darkmode else plt.cm.Dark2(i % 8),
            )
            
        plt.title(f'{self.model_name.capitalize()} Cluster Plot ({self.n_clusters} clusters)', 
                color='white' if darkmode else 'black', pad=20)
        plt.xlabel('Principal Component 1', color='white' if darkmode else 'black')
        plt.ylabel('Principal Component 2', color='white' if darkmode else 'black')
        plt.tick_params(colors='white' if darkmode else 'black')
        
        legend = plt.legend(facecolor='black', labelcolor='white' if darkmode else 'black')
        plt.setp(legend.get_frame(), color='black' if darkmode else 'white')
        
        plt.tight_layout()
        plt.savefig(filepath, facecolor='black' if darkmode else 'white', bbox_inches='tight')
        plt.close()
        
        plt.style.use('default')