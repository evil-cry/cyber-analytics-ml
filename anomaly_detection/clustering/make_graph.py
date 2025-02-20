# AI Usage Statement
# Tools Used: gpt-4o
# - Usage: Class iterator generation 
# - Verification: Class was successfuly iterated upon
# Prohibited Use Compliance: Confirmed

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

class Plot:
    def __init__(self, xlabel:str, ylabel:str, decorations:dict={-1:"#FF0000", 0:"#00FF00"}):
        self.graph_points = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.decorations = decorations

    def __iadd__(self, other):
        self.graph_points.append(other)
    
    # This was made using GPT-4o
    class PlotIterator:
        def __init__(self, graph_points):
            self._graph_points = graph_points
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self._graph_points):
                result = self._graph_points[self._index]
                self._index += 1
                return result
            else:
                raise StopIteration

    def __iter__(self):
        return self.PlotIterator(self.graph_points)
    
    def draw(self):
        pass


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