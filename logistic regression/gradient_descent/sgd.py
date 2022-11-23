import random
import time
from gd import *
from gd import LogisticRegressionClassifer as BaseLogistic


class LogisticRegressionClassifer(BaseLogistic):
    def stoch_gradient_ascent(self, X, y, learning_rate=0.01, epochs=1000):
        """
            stoch gradient ascent
            z = X*W     shape: (m, n)*(n, 1) = (m, 1)
        """
        X = np.matrix(X)   # (100, 3)
        # y = np.matrix(y)   # (1, 100)
        m, n = X.shape

        W = np.matrix(np.random.normal(size=(n, 1)))
        cost = []

        start = time.time()
        for i in range(epochs):
            # Randomly selected samples, batchsize = 1
            data_indices = list(range(m))
            random.shuffle(data_indices)

            for j, idx in enumerate(data_indices):
                data, label = X[idx], y[idx]
                y_hat = self.sigmoid(data * W)
                error = y_hat - label
                # 梯度下降
                W -= (1/m) * learning_rate * data.T * error
                cost.append(W.T.tolist()[0])

        end = time.time()
        print(f"sgd:epoch={epochs}, learning_rate={learning_rate}, 消耗的时间是：{end-start}")
        self.W = W
        return W, np.array(cost)


if __name__ == "__main__":
    data_file = "test.txt"
    file_name = "./snapshots_sgd"

    clf = LogisticRegressionClassifer()
    dataset, labels = load_data_logistic(data_file)
    W, cost = clf.stoch_gradient_ascent(dataset, labels)
    m, n = cost.shape

    # show data
    for i, w in enumerate(cost):
        if i % (m//10) == 0:
            print('sgd_{}.png saved'.format(i))
            snapshot(w.tolist(), dataset, labels, file_name, 'sgd_{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'W_{}'.format(i)
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(cost[:, i], label=label)
        ax.legend()

    fig.savefig('W_log_sgd.png')