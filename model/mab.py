import numpy as np

class MultiArmedBandit(object):

    def __init__(self, nb_input_features, nb_arms):
        self.nb_input_features = nb_input_features
        self.nb_arms = nb_arms
        self.arms = np.arange(self.nb_arms)
        


    def predict(self, features):
        """ Placeholder for predict """
        pass


class FixedDoseBandit(MultiArmedBandit):

    def __init__(self):
        super().__init__(0, 3)

    def predict(self, features):
        assert len(features) == self.nb_input_features
        return self.arms[1] # middle bucket, 5mg/day

class LinearClinicalBandit(MultiArmedBandit):

    def __init__(self):
        self.beta = np.array([4.0376, -0.2546, 0.0118, 0.0134, -0.6752, 0.406, 0.0443, 1.2799, -0.5695])
        super().__init__(3)

    def predict(self, features):
        assert len(features) == self.nb_input_features
        features = np.array([1].extend(features)) # add a bias
        return features.dot(self.beta) ** 2
        

if __name__ == "__main__":
    pass