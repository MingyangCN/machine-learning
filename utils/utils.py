import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_data_logistic(filename):
    dataset, labels = [], []
    with open(filename, 'r') as file:
        for line in file:
            # strip(): 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            # split(' '): 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串。
            split_line = [float(i) for i in line.strip().split('\t')]
            data, label = [1.0] + split_line[: -1], split_line[-1]
            dataset.append(data)
            labels.append(label)

    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels


def snapshot(W, dataset, labels, picture_name):
    """
        绘制分割线图
    """
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')

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

    x = [-4.0, 3.0]
    y = [get_y(i, W) for i in x]

    plt.plot(x, y, linewidth=2, color='#FB4A42')

    pic_name = './snapshots/{}'.format(picture_name)
    fig.savefig(pic_name)
    plt.close(fig)
    



