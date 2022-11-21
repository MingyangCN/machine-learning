import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression_grad_ascent import LogisticRegressionClassifer as BaseLogistic
from utils.utils import *


class LogisticRegressionClassifer(BaseLogistic):
    def stoch_gradient_ascent(self, X, y, learning_rate=0.01, epochs=10000):
        """
            stoch gradient ascent
            z = X*W     shape: (m, n)*(n, 1) = (m, 1)
        """
        X = np.matrix(X)   # (100, 3)
        # y = np.matrix(y)   # (1, 100)
        m, n = X.shape

        W = np.matrix(np.random.normal(size=(n, 1)))
        cost = []

        for i in range(epochs):
            # Randomly selected samples, batchsize = 1
            data_indices = list(range(m))
            random.shuffle(data_indices)

            for j, idx in enumerate(data_indices):
                data, label = X[idx], y[idx]
                y_hat = self.sigmoid(data * W)
                error = label - y_hat
                W += (1/m) * learning_rate * data.T * error

                cost.append(W.T.tolist()[0])

        self.W = W
        return W, np.array(cost)


if __name__ == "__main__":
    clf = LogisticRegressionClassifer()
    dataset, labels = load_data_logistic("testSet.txt")
    W, cost = clf.stoch_gradient_ascent(dataset, labels)
    m, n = cost.shape

    # show data
    for i, w in enumerate(cost):
        if i % (m//10) == 0:
            print('sgd_{}.png saved'.format(i))
            snapshot(w.tolist(), dataset, labels, 'sgd_{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'w{}'.format(i)
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(cost[:, i], label=label)
        ax.legend()

    fig.savefig('stoch_grad_ascent_params.png')







