import algorithms

def evaluate_instance(model, dimensions, params, data):
    '''
    Evaluate a model. 
    Score is based on lowering False Positive Rate.
    If False Negatives is >0, a heavy penalty is added.
    '''

    classes = {'kmeans': algorithms.KMeans, 'dbscan': algorithms.DBSCAN}
    ModelClass = classes.get(model)
    if not ModelClass:
        raise ValueError("Invalid model.")
    
    params_str = "_".join(f"{k}={v}" for k, v in params.items())
    graph_path = f"anomaly_detection/graphs/{model}/{params_str}.png"
    
    model = ModelClass(data, dimensions, 2, params)
    model.draw(graph_path)
    
    # Apply a heavy penalty if any false negatives
    penalty = 1000 if model.FN > 0 else 0
    score = model.fpr * 100 + penalty
    return (score, params, graph_path)

def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    k_means = algorithms.K_Means(data, 16, 2, {'k':2})
    dbscan = algorithms.DBSCAN(data, 16, 2, {'min':20})

    k_means.draw('anomaly_detection/graphs/kmeans.png')
    dbscan.draw('anomaly_detection/graphs/dbscan.png')

if __name__ == "__main__":
    main()