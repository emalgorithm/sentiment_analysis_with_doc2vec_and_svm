import numpy as np
import math
import scipy


def sign_test(y1_pred, y2_pred, y_test):
    # plus counts the number of times model1 beats model2
    plus = 0
    # minus counts the number of times model2 beats model1
    minus = 0
    # nul counts the number of times model1 and model2 tie
    null = 0

    for i in range(len(y_test)):
        correct1 = 1 if y1_pred[i] == y_test[i] else 0
        correct2 = 1 if y2_pred[i] == y_test[i] else 0

        if correct1 > correct2:
            plus += 1
        elif correct2 > correct1:
            minus += 1
        else:
            null += 1

    # If we have too many datapoints, than our custom method overflows
    # Therefore, in that case we use scipy function
    if len(y_test) > 500:
        return scipy.stats.binom_test(plus + null // 2, len(y_test))

    p = 0
    N = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    q = 0.5

    for i in range(k):
        p += 2 * (q ** i) * ((1 - q) ** (N - i)) * (
                    math.factorial(N) / (math.factorial(i) * math.factorial(N - i)))

    print("p: {}".format(p))


def permutation_test(y1, y2, y_true, samples=5000):
    y1 = [1 if y1[i] == y_true[i] else 0 for i in range(len(y_true))]
    y2 = [1 if y2[i] == y_true[i] else 0 for i in range(len(y_true))]

    original_difference = np.abs(np.mean(y1) - np.mean(y2))
    greater_samples = 0

    for i in range(samples):
        ps = [0.5 for j in range(len(y1))]
        flips = np.random.binomial(1, p=ps)
        y1_t = [y1[j] if flips[j] == 0 else y2[j] for j in range(len(y1))]
        y2_t = [y2[j] if flips[j] == 0 else y1[j] for j in range(len(y1))]
        diff = np.abs(np.mean(y1_t) - np.mean(y2_t))
        if diff >= original_difference:
            greater_samples += 1

    print("p: {}".format(greater_samples / samples))