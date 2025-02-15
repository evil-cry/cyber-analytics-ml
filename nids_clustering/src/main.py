import algorithms

def main():
    data = [
        "nids_clustering\\corpus\\KDD99\\testing_normal.npy", 
        "nids_clustering\\corpus\\KDD99\\testing_attack.npy", 
        "nids_clustering\\corpus\\KDD99\\training_normal.npy"
    ]

    # k_means = algorithms.K_means(data)
    dbscan = algorithms.DBSCAN(data)


if __name__ == "__main__":
    main()