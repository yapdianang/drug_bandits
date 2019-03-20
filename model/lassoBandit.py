import numpy as np

from sklearn.linear_model import Lasso
import losses
from utils import DataStream, get_arm_from_bucket_name
from tqdm import tqdm

seed = 234
# np.random.seed(seed)

class LASSOBandit(object):
    """ Implemented according to:
        http://web.stanford.edu/~bayati/papers/lassoBandit.pdf
    """

    def __init__(self, q, h, lambda1, lambda2, nb_feature_dims, K=3):
        """ Params:
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
        self.T = {i : [] for i in range(1, self.K+1)}
        self.generate_T_indices()
        for arm in self.T:
            print("sample T-indices for arm %d:" % arm)
            print(self.T[arm][:40])
            print('-' * 40)
        
        # Overall Sample History:
        #   For every arm i, the set of timesteps at which we acted by choosing arm i
        self.S = {i : [] for i in range(1, self.K+1)}
        
        # There's two beta's (learned param vectors) for every arm
        self.forced_sample_betas = np.zeros((self.K+1, self.d,), dtype=np.float64)
        self.forced_sample_bias = np.zeros((self.K+1, ), dtype=np.float64)
        self.all_sample_betas    = np.zeros((self.K+1, self.d,), dtype=np.float64)
        self.all_sample_bias = np.zeros((self.K+1, ), dtype=np.float64)

        self.forced_params = {}
        self.all_params = {}
        for i in range(1, self.K+1):
            self.forced_params[i] = Lasso(fit_intercept=False, alpha=(self.lambda1/2), max_iter=10000)
            # self.forced_params[i].coeff_ = np.zeros((self.d,))
            self.all_params[i]    = Lasso(fit_intercept=False, alpha=(self.lambda2/2), max_iter=10000)
            # self.all_params[i].coeff_ = np.zeros((self.d,))

        # These are the LASSO estimators that will be in charge of updating betas
        #   at every timestep.
        # lambda is halved, to accomodate for implementation vs. paper details

        # Dummy object to take up index 0, since we are 1-indexing
        self.observed_history_x = [None]
        self.observed_history_y = [None]

        # Initialize forced-sample sets.
        #   Generates timestep indices at which we will force a certain arm to be sampled,
        #   regardless of the features (i.e. covariates).
        #   (Equation (2), page 15 in the paper)
        

    def generate_T_indices(self):
        # The iteration over n in {0, 1, 2, ...} goes ad infinitum.
        #   With ~5500 data points, 15 is more than enough. 
        #   This gives us indices for T up to ~ 50K * q.
        sufficient_upper_n = 100
        for i in range(1, self.K+1):
            self.T[i] = [(2**n-1) * self.K * self.q + j \
                            for n in range(sufficient_upper_n) \
                            for j in range(self.q*(i-1) + 1, self.q*i + 1)
                        ]
        # NOTE:
        #  The indices are mathematically constructed to be mutually exclusive between arms!
        #  No two arms will have collisions in their respective forced-sampling indices.
        

    def access_indices_in_list(self, target_list, indices):
        return [target_list[i] for i in indices]


    def _get_action(self, timestep, x_features, training=False):
        """ INTERNAL FUNCTION. Used by |self.predict|.
            Params:
                timestep: non-negative index for timestep in the online prediction setting.
                x_features: numpy array (self.d,) containing features for the current patient.

            Returns:
                non-negative integer i, in [0, 1, ..., K-1], the arm we pull.
        """
        # If this timestep is one designated for a forced sampling of some arm, then do it
        if training:
            for i in self.T:
                if timestep in self.T[i]:
                    return i
        
        # FORCED-sample "filter for the seemingly-good arms"
        # Use the learned forced-sample parameters (self.forced_sample_betas)
        #   to estimate reward for each arm, and filter for the best handful
        estimated_rewards = [self.forced_sample_betas[i].dot(x_features.squeeze()) for i in range(1, self.K+1)]

        max_reward = max(estimated_rewards)
        arms_passing_threshold = [i+1 for i, reward in enumerate(estimated_rewards) if reward > max_reward - self.h / 2]
        # print("The arms passing threshold:", arms_passing_threshold)

        # ALL-sample "select the best one"
        # Use the learned all-sample parameters (self.all_sample_betas)
        #   to select one arm, based on the argmax.
        estimated_rewards = [(i, self.all_sample_betas[i].dot(x_features.squeeze())) for i in arms_passing_threshold]
        estimated_rewards = sorted(estimated_rewards, reverse=True, key=lambda tup: tup[1])
        # print("estimated_rewards:", estimated_rewards)
        selected_arm = estimated_rewards[0][0]  # Get the arm associated with highest estimated reward (with all_params)
        return selected_arm


    def predict(self, timestep, x_features, ground_truth_action=None, training=False):
        """ 
        Predicts the next action, can be used in training or testing. Calls self._get_action.
        """
        # |x_features| needs to be 2D (nb_samples, nb_features)
        augmented_x_features = x_features
        if len(x_features.shape) == 1:
            augmented_x_features = x_features[np.newaxis, :]
            # print("x_features shape:", x_features.shape)

        selected_arm = self._get_action(timestep, augmented_x_features, training=training)

        if not training:
            return selected_arm

        self.observed_history_x.append(x_features)
        self.observed_history_y.append(ground_truth_action)

        # Update self.S and self.lambda2, used to recompute self.all_sample_betas
        self.S[selected_arm].append(timestep)
        self.lambda2 = self.lambda2_initial * np.sqrt((np.log(timestep) + np.log(self.d)) / timestep)

        # recompute betas after updates
        # update forced sample beta update
        # print ("np.asarray(self.observed_history_x shape: {}".format(np.asarray(self.observed_history_x).shape))
        # print ("np.asarray(self.observed_history_y shape: {}".format(np.asarray(self.observed_history_y).shape))

        # Only the selected arm will potentially have any changes in their params
        T_up_til_now_indices = [ts for ts in self.T[selected_arm] if ts <= timestep]
        np_history_x = self.access_indices_in_list(self.observed_history_x, T_up_til_now_indices)
        np_history_y = self.access_indices_in_list(self.observed_history_y, T_up_til_now_indices)
        self.forced_params[selected_arm].fit(np_history_x, np_history_y)
        self.forced_sample_betas[selected_arm] = self.forced_params[selected_arm].coef_
        #self.forced_sample_bias[selected_arm] = fit_forced.intercept_


        for arm in range(1, self.K+1):
            lasso = self.all_params[arm]
            lasso.set_params(alpha = self.lambda2/2)  # update lambda2 for this timestep
            np_history_x =  self.access_indices_in_list(self.observed_history_x, self.S[arm])
            np_history_y = self.access_indices_in_list(self.observed_history_y, self.S[arm])
            if len(np_history_x) == 0:
                # There are no samples for this arm yet, cannot fit
                continue
            lasso.fit(np_history_x, np_history_y)
            self.all_sample_betas[arm] = lasso.coef_
            # self.all_sample_bias[arm] = fit_all.intercept_

        return selected_arm

    def evaluate(self, ds, mode='normal'):
        """
        Similar to LinUCB: every k iteration, evaluate the following:
            Accuracy: this is done from fresh every evaluation, i.e. accuracy at timestep 500 shouldn't depend on accuracy on timestep 250
            Regret: This is done cumulatively.

            This is the evaluation loop. A subroutine of the training loop, similar code.
        """
        all_actions = 0
        nb_correct = 0
        total_regret = 0.
        # run this on the test set
        actions = ['low', 'medium', 'high']
        for i, (features, ground_truth_action_name, real_dosage) in enumerate(zip(ds.table_test, ds.ground_truth_test, ds.dosage_test)):
            timestep = i+1  # Start timesteps 1-indexed
            ground_truth_action = get_arm_from_bucket_name(ground_truth_action_name)
            best_action = self.predict(timestep, features, ground_truth_action=ground_truth_action, training=False)
            reward = losses.calculate_reward(best_action, ground_truth_action, real_dosage, mode=mode)
            ############## print("gta %d, selected %d, reward %d" % (ground_truth_action, best_action, reward))
            total_regret += 0 - reward
            all_actions += 1
            nb_correct += 1. if (best_action == ground_truth_action) else 0.

        return (nb_correct/all_actions), total_regret
            





################################################################################


if __name__ == "__main__":

    # I made all these up. Please supply real values that work. --> Perhaps use argparse?
    q = 12
    h = 2
    lambda1 = 0.1
    lambda2 = 0.1

    mode = 'normal'
    validation_iters = 200

    ds = DataStream("../data/warfarin.csv", seed=seed)
    lasso_bandit = LASSOBandit(q, h, lambda1, lambda2, ds.feature_dim)
    print ("Number of features:  {}".format(ds.feature_dim))

    eval_acc_history = []
    eval_regret_history = []

    # Training Loop
    #   The paper assumes timesteps start at 1.
    for i, (features, ground_truth_action_name, real_dosage) in enumerate(tqdm(ds)): # Training Loop
        # Timesteps need to be 1-indexed. We use a dummy iteration index i, and set timestep=i+1.
        timestep = i+1
        ground_truth_action = get_arm_from_bucket_name(ground_truth_action_name)

        best_action = lasso_bandit.predict(timestep, features, ground_truth_action=ground_truth_action, training=True)
        # print (selected_arm, get_arm_from_bucket_name(ground_truth_action))
        training_reward = losses.calculate_reward(best_action, ground_truth_action, real_dosage, mode)
        # print (training_reward)
        training_regret = 0. - training_reward  # Unused rn

        # Every so often during the online training, validate against some val set
        #   Currently: val set = entire training set w/ ground truth
        if timestep % validation_iters == 0:
            new_ds = DataStream("../data/warfarin.csv")
            eval_accuracy, eval_regret = lasso_bandit.evaluate(new_ds, mode)
            eval_acc_history.append(eval_accuracy)
            eval_regret_history.append(eval_regret) # not cumulative so far
            print ("Accuracy at iter {}: {}".format(timestep, eval_accuracy))

    # plot 
