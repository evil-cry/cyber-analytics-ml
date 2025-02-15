import clustering

def main():
    data = [
        "anomaly_detection/corpus/KDD99/training_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_normal.npy", 
        "anomaly_detection/corpus/KDD99/testing_attack.npy", 
    ]

    # k_means = algorithms.K_means(data)
    dbscan = clustering.DBSCAN(data)


if __name__ == "__main__":
    main()