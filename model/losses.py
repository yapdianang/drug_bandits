import numpy as np

# General function outside LassoBandit class to calculate reward.
def calculate_reward(y_hat, y, real_dosage, mode='normal'):

    if mode == 'normal':
        return 0 if y_hat == y else -1

    # DEPRECATED dont use
    elif mode == 'mse':
        return -((y_hat - y)**2)

    elif mode == 'harsh':
        output = -(np.abs(y_hat - y).astype(float))  # 0, -1, or -2
        assert output in [0, -1, -2]
        return output

    elif mode == 'real':
        if y == 0:
            val = 1.5
        elif y == 1:
            val = 5
        else:
            val = 9
        return -np.abs(val - real_dosage)

    else:
        raise ValueError("Mode is not defined. Please select either one of 'binary', 'mse', 'harsh', 'real'")