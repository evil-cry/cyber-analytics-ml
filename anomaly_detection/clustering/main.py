import algorithms

def evaluate_instance(model_name, dimensions, params, data):
    '''
    Evaluate a model. 
    Score is based on lowering False Positive Rate.
    If False Negatives is >0, a heavy penalty is added.
    '''

    classes = {'kmeans': algorithms.K_Means, 'dbscan': algorithms.DBSCAN}
    ModelClass = classes.get(model_name)
    if not ModelClass:
        raise ValueError("Invalid model.")
    
    params_str = "_".join(f"{k}={v}" for k, v in params.items())
    graph_path = f"anomaly_detection/graphs/{model_name}/{params_str}.png"
    
    model = ModelClass(data, dimensions, 2, params)
    model.draw(graph_path)
    
    # Apply a heavy penalty if false negatives > 0
    penalty = 1000 if model.FN > 0 else 0
    score = model.fpr * 100 + penalty
    
    with open(f'anomaly_detection/graphs/{model_name}/evals.txt', 'a') as f:
        f.write(model.print_score(score))

    return (score, params, graph_path)

def find_best(data):
    '''
    Find the best model. 
    The parameters are very different and may be correlated to each other, so values need to be changed manually.  
    '''
    
    def dbscan_search():
        '''
        Search for the best parameters for DBSCAN. 
        DBScan's parameters are more closely correlated than KMeans' parameters, so do a nested loop
        '''
        
        # Best 
        # e = 0.0146
        # min = 2
        for e in range(170, 200, 1):
            for min in range(2, 3, 1):
                evaluate_instance('dbscan', 16, {'e': e / 10000, 'min': min}, data)
    
    def kmeans_search():
        '''
        Search for the best parameters for KMeans.
        These are not as correlated as dbscan's, so check them separately 
        '''

        # Best
        # k = 79
        # tolerance = 0.0001
        # max = 100
        # threshold = 95
        for d in range(1, 42, 1):
            evaluate_instance('kmeans', d, {}, data)

    #dbscan_search()
    kmeans_search()

def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    #find_best(data)

    k_means = algorithms.K_Means(data, 16, 2)
    #dbscan = algorithms.DBSCAN(data, 16, 2)

    k_means.draw('anomaly_detection/graphs/kmeans.png')
    #dbscan.draw('anomaly_detection/graphs/dbscan.png')

if __name__ == "__main__":
    main()