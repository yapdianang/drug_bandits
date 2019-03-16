import numpy as np
from utils import DataStream

class LASSOBandit(object):
    """ Implemented according to:
        http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
    """

    def __init__(self, q, h, lambda1, lambda2, nb_feature_dims, K=3):
        """
            Params:
                q is a positive integer.
                h is a positive real. Both are used to parametrize the LASSO Bandit model.
                lambda1 is used to regularize forced_sample_beta.
                lambda2 is used to regularize all_sample_beta.
        """
        self.q = q
        self.h = h
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda2_initial = lambda2  # This is used to recompute self.lambda2 on future timesteps
        self.d = nb_feature_dims
        self.K = K

        # Force-Sample Sets:
        #   For every arm i, the set of timesteps at which we will force-sample arm i
        self.T = {i : [] for i in range(self.K)}
        
        # Overall Sample History:
        #   For every arm i, the set of timesteps at which we acted by choosing arm i
        self.S = {i : [] for i in range(self.K)}
        
        # There's two beta's (learned param vectors) for every arm
        self.forced_sample_betas = np.zeros((self.K, self.d,), dtype=np.float64)
        self.all_sample_betas    = np.zeros((self.K, self.d,), dtype=np.float64)

        
        # Initialize forced-sample sets.
        #   Generates timestep indices at which we will force a certain arm to be sampled,
        #   regardless of the features (i.e. covariates).
        #   (Equation (2), page 15 in the paper)
        
        # The iteration over n in {0, 1, 2, ...} goes ad infinitum.
        #   With ~5500 data points, 15 is more than enough. 
        #   This gives us indices for T up to ~ 50K * q.
        sufficient_upper_n = 15
        for i in range(self.K):
            self.T[i] = [(2**n-1) * self.K * self.q + j \
                            for n in range(sufficient_upper_n) \
                            for j in range(self.q*(i-1) + 1, self.q*i + 1)
                        ]
        # NOTE:
        #  The indices are mathematically constructed to be mutually exclusive between arms!
        #  No two arms will have collisions in their respective forced-sampling indices.
        

    def _get_action(self, timestep, x_features):
        """ INTERNAL FUNCTION. Used by |self.predict|.
            Params:
                timestep: non-negative index for timestep in the online prediction setting.
                x_features: numpy array (self.d,) containing features for the current patient.

            Returns:
                non-negative integer i, in [0, 1, ..., K-1], the arm we pull.
        """
        # If this timestep is one designated for a forced sampling of some arm, then do it
        for i in self.T:
            if timestep in self.T[i]:
                return i
        
        # FORCED-sample "filter for the seemingly-good arms"
        # Use the learned forced-sample parameters (self.forced_sample_betas)
        #   to estimate reward for each arm, and filter for the best handful
        estimated_rewards = [x_features.dot(self.forced_sample_betas[i]) for i in range(self.K)]
        max_reward = max(estimated_rewards)
        arms_passing_threshold = [i for i, reward in enumerate(estimated_rewards) if reward > max_reward - self.h / 2]
        
        # ALL-sample "select the best one"
        # Use the learned all-sample parameters (self.all_sample_betas)
        #   to select one arm, based on the argmax.

        ##### To check: should the next line be (estimated_rewards = [x_features.dot(self.all_sample_betas[i]) for i in arms_passing_threshold])? #####
        estimated_rewards = [x_features.dot(self.all_sample_betas[i]) for i in range(self.K)]
        selected_arm = np.argmax(estimated_rewards)
        return selected_arm


    def predict(self, timestep, x_features):
        selected_arm = self._get_action(timestep, x_features)

        # Update self.S and self.lambda2, used to recompute self.all_sample_betas
        self.S[selected_arm].append(timestep)
        self.lambda2 = self.lambda2_initial * np.sqrt(np.log(timestep * self.d) / timestep)

        # TODO(ojwang): recompute betas after updates

        return selected_arm

        




################################################################################


if __name__ == "__main__":

    # I made all these up. Please supply real values that work.
    q = 1
    h = 0.5
    lambda1 = 0.25
    lambda2 = 0.125
    nb_feature_dims = 10

    lasso_bandit = LASSOBandit(q, h, lambda1, lambda2, nb_feature_dims)
    ds = DataStream("myroot/mydir/my_csv_file_name.csv")

    # TODO(ojwang): 1-indexing
    #   The paper assumes timesteps start at 1.
    #   The paper assumes actions/arms are enumerated [1, 2, ..., K]
    for timestep, (features, ground_truth_action, real_dosage) in enumerate(ds):
        # Do something
        pass
        # lasso_bandit.predict(timestep)


