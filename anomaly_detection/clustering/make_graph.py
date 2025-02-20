import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import randint

interactive = plt.get_backend()
matplotlib.use('Agg')

class Plot:
    def __init__(self, dimensions):
        self.graph_points = {}
        self.title = ''
        self.xlabel = ''
        self.ylabel = ''
        self.colors = {-1:"#FF0000", 0:"#00FF00"}
        self.labels = {-1: "Anomaly", 0: "Normal"}
        self.path = ''
        self.dimensions = dimensions

    def configure(self, xlabel, ylabel, title = '', colors = None, labels = None, path = None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        if colors:
            self.colors = colors
        if labels:
            self.labels = labels
        if path:
            self.path = path

    def __iadd__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            coordinate, tag = other
            if tag not in self.graph_points:
                self.graph_points[tag] = []
            self.graph_points[tag].append(coordinate)
            return self
        else:
            raise ValueError("Can only add tuple(coordinate, tag) to Plot.")

    def draw(self, show):
        for tag, coordinates in self.graph_points.items():
            x, y = zip(*coordinates)

            if tag in self.colors:
                tag_color = self.colors[tag]
            else:
                tag_color = randint(0, 4294967296) # FFFFFF
                tag_color = f'#{tag_color:x}'

            if tag in self.labels:
                tag_label = self.labels[tag]
            else:
                tag_label = f'Mystery'

            if show:
                matplotlib.use(interactive)

            plt.scatter(x=x, y=y, c=tag_color, label=tag_label)
            plt.xlabel = self.xlabel
            plt.ylabel = self.ylabel
            plt.title(self.title)

        if show:
            plt.legend()
            plt.show()
        if self.path:
            plt.savefig(self.path)
        matplotlib.use('Agg')

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