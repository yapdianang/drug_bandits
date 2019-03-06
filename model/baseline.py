import numpy as np
from mab import MultiArmedBandit

class FixedDoseBandit(MultiArmedBandit):

    def __init__(self):
        super().__init__(0)

    def estimate_dosage(self, features):
        return 5 # dosage, mg/day


class LinearClinicalBandit(MultiArmedBandit):

    def __init__(self):
        self.beta = np.array([4.0376, -0.2546, 0.0118, 0.0134, -0.6752, 0.406, 0.0443, 1.2799, -0.5695])
        super().__init__(len(self.beta))

    def extract_features(self, df_row):
        """ @ TODO: Justin Xu
        """
        pass

    def estimate_dosage(self, features):
        features = np.array([1].extend(features)) # add a bias
        return (features.dot(self.beta) ** 2) / 7
        

if __name__ == "__main__":
    pass