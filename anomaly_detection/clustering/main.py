import algorithms

def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    k_means = algorithms.K_Means(data, 16, 2, {'k':2})
    dbscan = algorithms.DBSCAN(data, 16, 2, {'min':20})

    k_means.draw(True)
    dbscan.draw(True)

if __name__ == "__main__":
    main()