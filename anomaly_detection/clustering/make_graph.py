#%%
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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