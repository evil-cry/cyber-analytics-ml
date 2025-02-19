import anomaly_detection.clustering.algorithms as algorithms


def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    k_means = algorithms.K_Means(data)
    dbscan = algorithms.DBSCAN(data)


if __name__ == "__main__":
    main()