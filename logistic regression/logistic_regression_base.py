"""
Author:Mingyang Wu
Day:16.11.2022
Abstract: 逻辑回归，求最优决策边界
Tips:
    Z = w_0 + w_1*x_1 + w_2*x_2  ==> Z = xw  ==> shape (10 ,3) * (3, 1)  => (10, 1)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
        be used in logistic regression
        filename: filepath
        s.isspace() 判断字符串中是否包含空格
    """
    file = open(filename)

    X = []  # [1, x1, x2]
    y = []
    for line in file.readlines():
        # line = line.split(" ")  括号内是两个数据之间的分隔符
        line = line.strip().split(",")

        # float() argument must be a string or a number, not 'list'
        X.append([1, float(line[0]), float(line[1])])
        y.append([float(line[-1])])

    X_matrix = np.mat(X)
    y_matrix = np.mat(y)

    file.close()
    return X_matrix, y_matrix


def w_calc(xmat, ymat, lr=0.001, epochs=20000):
    # 服从正态分布-> W_init
    W = np.mat(np.random.randn(3, 1))

    for i in range(epochs):
        # W update
        H = 1/(1+np.exp(-xmat*W))
        dw = xmat.T*(H - ymat)  # (3,1)
        W -= lr*dw

    return W


def show_bound(xmat, ymat, W):
    plt.scatter(xmat[:, 1][ymat == 0].A, xmat[:, 2][ymat == 0].A)
    plt.scatter(xmat[:, 1][ymat == 1].A, xmat[:, 2][ymat == 1].A)
    # plt.show()

    w0 = W[0, 0]
    w1 = W[1, 0]
    w2 = W[2, 0]

    plotx1 = np.arange(1, 7, 0.01)  # x1的取值
    plotx2 = -(w0/ w1) - w1/ w2*plotx1
    plt.plot(plotx1, plotx2, c='r', label="decision boundary")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filename = "./data.txt"
    xmat, ymat = load_data(filename)
    print(xmat.shape, ymat.shape)
    W = w_calc(xmat, ymat)
    print('W:', W)

    # show data result
    show_bound(xmat, ymat, W)



