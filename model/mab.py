class MultiArmedBandit(object):

    def __init__(self, nb_input_features, nb_arms=3):
        self.nb_input_features = nb_input_features
        self.nb_arms = nb_arms
        
    def extract_features(self, df_row):
        """ Placeholder for extract_features """
        pass

    def estimate_dosage(self, features):
        """ Placeholder for estimate_dosage """
        pass

    def predict(self, features):
        """ General function for predicting 1 of 3 buckets """
        assert len(features) == self.nb_input_features
        prediction = self._estimate_dosage(features)
        arm = 0
        if prediction > 3:
            arm = 1
        elif prediction > 7:
            arm = 2
        return arm
