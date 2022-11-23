#coding=utf-8
import numpy as np
import random

#下面实现的是批量梯度下降法
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()                             #得到它的转置
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # print loss
        gradient = np.dot(xTrains, loss) / m             #对所有的样本进行求和，然后除以样本数
        theta = theta - alpha * gradient
    return theta

#下面实现的是随机梯度下降法
def StochasticGradientDescent(x, y, theta, alpha, m, maxIterations):
    data = []
    for i in range(10):
        data.append(i)
    xTrains = x.transpose()     #变成3*10，没一列代表一个训练样本
    # 这里随机挑选一个进行更新点进行即可（不用像上面一样全部考虑）
    for i in range(0,maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y                   #注意这里有10个样本的，我下面随机抽取一个进行更新即可
        index = random.sample(data,1)           #任意选取一个样本点，得到它的下标,便于下面找到xTrains的对应列
        index1 = index[0]                       #因为回来的时候是list，我要取出变成int，更好解释
        gradient = loss[index1]*x[index1]       #只取这一个点进行更新计算
        theta = theta - alpha * gradient.T
    return theta

def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))                     #在这个例子中，是第三列放1
    xTest[:, :-1] = x                             #前俩列与x相同
    res = np.dot(xTest, theta)                    #预测这个结果
    return res

if __name__ == "__main__":
    trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
    trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])
    m, n = np.shape(trainData)
    theta = np.ones(n)
    alpha = 0.1
    maxIteration = 5000
    #下面返回的theta就是学到的theta
    theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
    print(f"theta = {theta}")
    x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
    print(predict(x, theta))
    theta = StochasticGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
    print(f"theta = {theta}")
    x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
    print(predict(x, theta))
    #yes,is the code
