import numpy as np
from sklearn import decomposition

class _Algorithm():

    def __init__(self, corpus) -> None:
        self.testing_normal_data = _Algorithm.load_corpus(corpus[0])
        self.testing_anomaly_data = _Algorithm.load_corpus(corpus[1])
        self.training_normal_data = _Algorithm.load_corpus(corpus[2])

    @staticmethod
    def load_corpus(path: str) -> np.array:
        data = np.load(path)
        data = _Algorithm.reduce(data)

        return data

    @staticmethod
    def reduce(data: np.array) -> np.array:
        print(f"Original shape: {data.shape}")

        pca = decomposition.PCA(n_components=2)

        fit_data = pca.fit_transform(data)

        print(f"Transformed shape: {fit_data.shape}")

        return fit_data

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
