import numpy as np
import pandas as pd
from utils import get_bucket, get_accuracy, get_features_and_dosage
from mab import MultiArmedBandit


class FixedDoseBaseline(MultiArmedBandit):

    def __init__(self):
        super().__init__(0)

    def __repr__(self):
        return 'fixed dose'

    def estimate_dosage(self, features):
        return 'medium' # dosage, mg/day


class LinearClinicalBaseline(MultiArmedBandit):

    def __init__(self):
        self.beta = np.array([4.0376, -0.2546, 0.0118, 0.0134, -0.6752, 0.406, 0.0443, 1.2799, -0.5695])
        super().__init__(len(self.beta))

    def __repr__(self):
        return 'linear baseline'

    def extract_features(self, df_row):
        """ @ TODO: Justin Xu
        this should work without problems
        """
        pass

    def estimate_dosage(self, features):
        dosage = (features.dot(self.beta) ** 2) / 7
        return get_bucket(dosage)

def predict_dosages(baseline, features):
    return features.apply(bandit.estimate_dosage, axis=1)

if __name__ == "__main__":
    # df = pd.read_csv('../data/clinical_dosing_features.csv')    
    # features = df[['Bias', 'Age in decades', 'Height (cm)', 'Weight (kg)', 'Asian', 'Black or African American', 
    #            'Missing or Mixed Race','Enzyme inducer status', 'Amiodarone status']]
    # true_dosages = df['dosage_bucket']

    df, features, true_dosages = get_features_and_dosage('../data/clinical_dosing_features.csv')

    def get_dosage_bucket(x):
        dosage = x.dot(beta)**2 / 7
        return get_bucket(dosage)

    fixed_baseline = FixedDoseBaseline()
    linear_baseline = LinearClinicalBaseline()
    baselines = [fixed_baseline, linear_baseline]

    for baseline in baselines:
        dosage_buckets = features.apply(baseline.estimate_dosage, axis=1)
        accuracy = get_accuracy(dosage_buckets, true_dosages)
        print('accuracy of {} is {}'.format(baseline, accuracy))
