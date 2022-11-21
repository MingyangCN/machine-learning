from utils.utils import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class LogisticRegressionClassifer(object):
    """
        梯度上升法
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def gradient_ascent(self, X, y, learning_rate=0.001, epochs=1000):
        """
            使用梯度上升优化Logistic回归模型参数
            z = XW + b   (m, 1)  = (m, n) * (n, 1)
            (W.T)X

        :param X: 数据特征矩阵
        :type X: MxN numpy matrix

        :param y: 数据集对应的类型向量  -> labels
        :type y: Nx1 numpy matrix
        """
        X = np.matrix(X)  # (m, n)
        y = np.matrix(y).reshape(-1, 1)  # (m, 1)  转化成1列
        m, n = X.shape

        # init weights
        W = np.random.normal(size=(n, 1))  # (n, 1)
        cost = []

        for i in range(epochs):
            y_hat = self.sigmoid(X * W)
            error = y - y_hat  # (m, 1)
            W += learning_rate * X.T * error  # ?

            # reshape(1, -1) 转化成1行
            cost.append(W.reshape(1, -1).tolist()[0])

        self.W = W
        return W, np.array(cost)

    def predict(self, X, W=None):
        """
            predict test_dataset
        """
        if W is None:
            W = self.W

        X = np.matrix(X)
        y_hat = X*W
        probability = self.sigmoid(y_hat.tolist()[0][0])

        return round(probability)


if __name__ == "__main__":
    data_file = "testSet.txt"
    # data = X.shape ==> (100, 3)

    log_reg = LogisticRegressionClassifer()
    X, y = load_data_logistic(data_file)
    W, cost = log_reg.gradient_ascent(X, y)
    m, n = X.shape

    for i in range(600):
        if i % (30) == 0:
            print('{}.png saved'.format(i))
            snapshot(cost[i].tolist(), X, y, '{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'W_{}'.format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(cost[:, i], label=label)
        ax.legend()

    fig.savefig('W_log_reg.png')
