from utils.utils import load_data_logistic

import numpy as np
import matplotlib.pyplot as plt


class perception(object):
    """
        单层感知机
    """
    def sign(self, x):
        """
            跃阶函数
        """
        if x >= 0:
            return 1
        else:
            return -1

    def train(self, X, y, learning_rate=0.001, epochs=1000):
        m, n = X.shape
        W = np.random.normal(size=(n, 1))

        for i in range(epochs):
            # 遍历每个数据的权重, 每个权值
            for data, label in zip(X, y):
                bias, x_1, x_2, y_label = data[0], data[1], data[2], label
                score = bias * W[0] + x_1 * W[1] + x_2 * W[2]

                if (y_label * self.sign(score)) <= 0:
                    W[0] += learning_rate * y_label               # bias
                    W[1] += learning_rate * y_label * x_1
                    W[2] += learning_rate * y_label * x_2
                    # print(W)
        return W


def normal(X):
    """
        标准化
    """
    # 均值
    u = np.mean(X, axis=0)
    # 方差
    v = np.std(X, axis=0)
    X = (X - u) / v

    return X



def plot_fig(X, y):
    plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='blue', marker='o', label='Positive')
    plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='red', marker='x', label='Negative')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')  # 函数的作用是给图像加图例
    plt.title('Original Data')
    plt.show()


def plot_points(X, y, w):
    b = w[0]
    plt.figure()
    x1 = np.linspace(-3, 3, 1000)
    x2 = (-b - w[1] * x1) / w[2]  # 化简 w1*x1 + w2*x2 + b =0 此时x2相当于竖轴坐标
    plt.plot(x1, x2, color='r', label='y1 data')

    plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], color='blue', marker='o', label='Positive')
    plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1], color='red', marker='x', label='Negative')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')  # 函数的作用是给图像加图例
    plt.title('Result')
    plt.show()


if __name__ =="__main__":
    path = "testSet.txt"
    dataset, labels = load_data_logistic(path)  # x[0] ==> bias  X加上偏置项

    X = normal(dataset[:, 1:3])
    y = labels
    plot_fig(X, y)

    X = np.hstack((np.ones((X.shape[0], 1)), X))  # x[0] ==> bias  X加上偏置项

    pla = perception()      # 感知机, 实例化
    w = pla.train(X, y)
    plot_points(X, y, w)


"""
1. 两者的损失函数有所不同，PLA针对误分类点到超平面的距离总和进行建模，LR使用交叉熵损失。
2. 两者的优化方法可以统一为GD \ SGD。

Logistic Regression 比 PLA 的优点之一在于对于激活函数的改进。
前者为sigmoid function，后者为step function。LR使得最终结果有了概率解释的能力（将结果限制在0-1之间），sigmoid为平滑函数（连续可导），能够得到更好的分类结果，而step function为分段函数，对于分类的结果处理比较粗糙，非0即1，而不是返回一个分类的概率。

`感知机和逻辑回归的区别`
1. 感知机不求概率，一般会选取一个分类边界，可能y>0就是A类别，y<0就是B类别。
2. 逻辑回归的损失函数由最大似然推导，力图使预测概率分布与真实概率分布接近。
    感知机的损失函数可能有多种方法，可能有多层感知机，
- 但他们本质的思想都是使预测的结果与真实结果误差更小，是函数拟合，是去求得分类超平面。
"""