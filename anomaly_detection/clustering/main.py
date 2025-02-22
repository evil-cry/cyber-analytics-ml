import algorithms
import os

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
    graph_path = f"anomaly_detection/graphs/{model_name}/{dimensions}/{params_str}.png"
    if not os.path.isdir(f'anomaly_detection/graphs/{model_name}/{dimensions}'):
        os.mkdir(f'anomaly_detection/graphs/{model_name}/{dimensions}')
    
    model = ModelClass(data, dimensions, 2, params)
    model.draw(graph_path)
    
    # Apply a heavy penalty if false negatives > 0
    penalty = 1000 if model.FN > 0 else 0
    score = model.fpr * 100 + penalty
    
    with open(f'anomaly_detection/graphs/{model_name}/evals.txt', 'a') as f:
        f.write(f'{(dimensions)} ' + model.print_score(score))

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
        for d in range(10, 31, 1):
            with open(f'anomaly_detection/graphs/{'dbscan'}/evals.txt', 'a') as f:
                f.write('\n')
            for e in range(100, 201, 10):
                evaluate_instance('dbscan', d, {'e': e / 10000}, data)
    
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
        for d in range(2, 35, 3):
            with open(f'anomaly_detection/graphs/{'kmeans'}/evals.txt', 'a') as f:
                f.write('\n')
            for k in range(40, 86, 5):
                for _ in range(5):
                    evaluate_instance('kmeans', d, {'k':k}, data)

    #dbscan_search()
    kmeans_search()

def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    find_best(data)

    # 18 dimensions was found to be the best for kmeans experimentally 
    #k_means = algorithms.K_Means(data, 18, 2)
    #dbscan = algorithms.DBSCAN(data, 18, 2)

    #k_means.draw('anomaly_detection/graphs/kmeans.png')
    #dbscan.draw('anomaly_detection/graphs/dbscan.png')

if __name__ == "__main__":
    main()