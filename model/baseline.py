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

def get_fixed_baseline():
    fixed_baseline = FixedDoseBaseline()

    # read in csv and get non na values
    df = pd.read_csv('../data/warfarin.csv')
    df = df.dropna(subset = ['Therapeutic Dose of Warfarin'])

    # get predicted buckets and true dosage buckets
    dosage_buckets = df.apply(fixed_baseline.estimate_dosage, axis=1)
    true_dosages = (df['Therapeutic Dose of Warfarin'] / 7).apply(get_bucket)

    accuracy = get_accuracy(dosage_buckets, true_dosages)
    print('accuracy of {} is {}'.format(fixed_baseline, accuracy))

def get_linear_baseline():
    df, features, true_dosages = get_features_and_dosage('../data/clinical_dosing_features.csv')

    # get predicted dosages
    linear_baseline = LinearClinicalBaseline()
    dosage_buckets = features.apply(linear_baseline.estimate_dosage, axis=1)

    accuracy = get_accuracy(dosage_buckets, true_dosages)
    print('accuracy of {} is {}'.format(linear_baseline, accuracy))

if __name__ == "__main__":
    get_fixed_baseline()
    get_linear_baseline()
  