import numpy as np

class MultiArmedBandit(object):

    def __init__(self, nb_input_features, nb_arms=3):
        self.nb_input_features = nb_input_features
        self.nb_arms = nb_arms
        
    def predict(self, features):
        """ Placeholder for predict """
        pass

    def output_arm(self, prediction):
        arm = 0
        if prediction > 3:
            arm = 1
        elif prediction > 7:
            arm = 2
        return arm

class FixedDoseBandit(MultiArmedBandit):

    def __init__(self):
        super().__init__(0)

    def predict(self, features):
        assert len(features) == self.nb_input_features
        prediction = 5
        return self.output_arm(prediction)


class LinearClinicalBandit(MultiArmedBandit):

    def __init__(self):
        self.beta = np.array([4.0376, -0.2546, 0.0118, 0.0134, -0.6752, 0.406, 0.0443, 1.2799, -0.5695])
        super().__init__(len(self.beta))


    def predict(self, features):
        assert len(features) == self.nb_input_features
        features = np.array([1].extend(features)) # add a bias
        prediction = features.dot(self.beta) ** 2
        return self.output_arm(prediction)
        
        

if __name__ == "__main__":
    pass