import numpy as np

class _Algorithm():

    def __init__(self, corpus) -> None:
        self.testing_normal_data = corpus[0]
        self.testing_anomaly_data = corpus[1]
        self.training_normal_data = corpus[2]

        self.testing_normal_data = np.load(self.training_normal_data)
        self.testing_anomaly_data = np.load(self.testing_anomaly_data)
        self.training_normal_data = np.load(self.training_normal_data)

        print(f"Testing data shape: {self.testing_normal_data.shape}")
        print(f"Anomaly data shape: {self.testing_anomaly_data.shape}")
        print(f"Training data shape: {self.training_normal_data.shape}")


    def train(self) -> None:
        raise NotImplementedError()


    def test(self) -> None:
        raise NotImplementedError()


    def evalute(self) -> None:
        raise NotImplementedError()


class K_means(_Algorithm):

    def __init__(self, corpus) -> None:
        super(K_means, self).__init__(corpus)

        self.name = "K Means"


class DBSCAN(_Algorithm):

    def __init__(self, *args, **kwargs) -> None:
        super(K_means, self).__init__(*args, **kwargs)

        self.name = "Density-based spatial clustering of applications with noise"
