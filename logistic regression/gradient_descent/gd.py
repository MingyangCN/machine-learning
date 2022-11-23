import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt


def load_data_logistic(filename):
    dataset, labels = [], []
    with open(filename, 'r') as file:
        for line in file:
            split_line = [float(i) for i in line.strip().split(',')]
            data, label = [1.0] + split_line[: -1], split_line[-1]
            dataset.append(data)
            labels.append(label)

    dataset = np.array(dataset)
    # 数据标准化
    u = np.mean(dataset[:, 1:3], axis=0)  # 方差
    v = np.std(dataset[:, 1:3], axis=0)  # 方差
    dataset[:, 1:3] = (dataset[:, 1:3] - u) / v

    labels = np.array(labels)

    return dataset, labels


class LogisticRegressionClassifer(object):
    """
        梯度下降法
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def gradient_descent(self, X, y, learning_rate=0.001, epochs=1000):
        """
            batch_gradient_descent
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

        W = np.random.normal(size=(n, 1))  # (n, 1) init weights
        cost = []

        start = time.time()
        for i in range(epochs):
            y_hat = self.sigmoid(X * W)
            error = y_hat - y  # (m, 1)
            W -= learning_rate * X.T * error
            # W -= learning_rate * X.T * error * (1/m)

            cost.append(W.reshape(1, -1).tolist()[0])  # reshape(1, -1) 转化成1行

        end = time.time()
        print(f"gd:epoch={epochs}, learning_rate={learning_rate}, 消耗的时间是：{end-start}")
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


def snapshot(W, dataset, labels, file_name, picture_name):
    """
        绘制分割线图
    """
    if not os.path.exists(file_name):
        os.mkdir(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pts = {}
    for data, label in zip(dataset.tolist(), labels.tolist()):
        pts.setdefault(label, [data]).append(data)

    for label, data in pts.items():
        data = np.array(data)
        plt.scatter(data[:, 1], data[:, 2], label=label, alpha=0.5)

    # 分割线绘制
    def get_y(x, w):
        w0, w1, w2 = w
        return (-w0 - w1*x)/w2

    x = [-3.0, 3.0]
    y = [get_y(i, W) for i in x]

    plt.plot(x, y, linewidth=2, color='#FB4A42')

    pic_name = './{}/{}'.format(file_name, picture_name)
    fig.savefig(pic_name)
    plt.close(fig)


if __name__ == "__main__":
    data_file = "test.txt"
    file_name = "./snapshots_bgd"

    log_reg = LogisticRegressionClassifer()
    X, y = load_data_logistic(data_file)
    W, cost = log_reg.gradient_descent(X, y)
    m, n = X.shape

    for i in range(600):
        if i % (50) == 0:
            print('bgd_{}.png saved'.format(i))
            snapshot(cost[i].tolist(), X, y, file_name,'bgd_{}.png'.format(i))

    fig = plt.figure()
    for i in range(n):
        label = 'W_{}'.format(i)
        ax = fig.add_subplot(n, 1, i+1)
        ax.plot(cost[:, i], label=label)
        ax.legend()

    fig.savefig('W_log_bgd.png')