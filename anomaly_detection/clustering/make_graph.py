import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint

matplotlib.use('Agg')

class Plot:
    def __init__(self, dimensions):
        self.graph_points = {}
        self.title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.size = 1
        self.colors = {'tp':"#FF0000", 'tn':"#00FF00", 'fp':"#0000FF", 'fn':'#000000'}
        self.labels = {'tp': "True Positive", 'tn': "True Negative", 'fp': "False Positive", 'fn':"False Negative"}
        self.dimensions = dimensions

    def configure(self, xlabel, ylabel, title = '', size = 1, colors = None, labels = None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.size = size
        if colors:
            self.colors = colors
        if labels:
            self.labels = labels

    def __iadd__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            coordinate, tag = other
            if tag not in self.graph_points:
                self.graph_points[tag] = []
            self.graph_points[tag].append(coordinate)
            return self
        else:
            raise ValueError("Can only add tuple(coordinate, tag) to Plot.")

    def draw(self, path):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        for tag, coordinates in self.graph_points.items():
            x, y = zip(*coordinates)

            if tag in self.colors:
                tag_color = self.colors[tag]
            else:
                tag_color = randint(0, 0xFFFFFF+1) # Generating a fully random color isn't the best idea for graphs on white background but this is a fallback anyway
                tag_color = f'#{tag_color:x}'

            if tag in self.labels:
                tag_label = self.labels[tag]
            else:
                tag_label = f'Mystery'

            plt.scatter(x=x, y=y, c=tag_color, label=tag_label, s=self.size, alpha=0.5, zorder=2)

        plt.savefig(path, dpi=1200)

def plot_eps(data: np.array, min_pts: int) -> None:
    '''
    Plot the k-distance graph to estimate the epsilon value
    '''
    
    print("Plotting distances")

    k_distances = np.sort(data, axis=1)[:, min_pts]
    k_distances = np.sort(k_distances)
    
    plt.plot(k_distances)
    plt.xlabel('Data points')
    plt.ylabel('Epsilon')
    plt.title('Elbow plot for epsilon estimation')
    plt.savefig('nids_clustering/graphs/epsilon.png')
    
    return k_distances