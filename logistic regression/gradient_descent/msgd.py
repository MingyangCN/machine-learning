import random
import time

from gd import *
from gd import LogisticRegressionClassifer as BaseLogistic


class LogisticRegressionClassifer(BaseLogistic):
    def mini_batch_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000, batch_size=16, shuffle=True):
        """
            mini batch stoch gradient ascent
            z = X*W     shape: (m, n)*(n, 1) = (m, 1)
        """
        X = np.matrix(X)   # (100, 3)
        m, n = X.shape

        W = np.matrix(np.random.normal(size=(n, 1)))
        cost = []

        start = time.time()
        for i in range(epochs):
            if shuffle == True:
                data_indices = list(range(m))
                random.shuffle(data_indices)
                data_indices = data_indices[:batch_size]

            for j, idx in enumerate(data_indices):

                data, label = X[idx], y[idx]
                y_hat = self.sigmoid(data * W)
                error = y_hat - label
                W -= (1/m) * learning_rate * data.T * error
                cost.append(W.T.tolist()[0])

        end = time.time()
        print(f"msgd:epoch={epochs}, learning_rate={learning_rate}, batchsize={batch_size}, 消耗的时间是：{end-start}")
        self.W = W
        return W, np.array(cost)


if __name__ == "__main__":
    data_file = "test.txt"
    file_name = "./snapshots_bsgd"

    log_reg = LogisticRegressionClassifer()
    dataset, labels = load_data_logistic(data_file)
    W, cost = log_reg.mini_batch_gradient_descent(dataset, labels)
    m, n = cost.shape

    # show data
    for i, w in enumerate(cost):
        if i % (m//10) == 0:
            print('msgd_{}.png saved'.format(i))
            snapshot(w.tolist(), dataset, labels, file_name, 'msgd_{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'W_{}'.format(i)
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(cost[:, i], label=label)
        ax.legend()

    fig.savefig('W_log_msgd.png')








