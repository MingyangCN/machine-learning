""" reference: https://blog.csdn.net/qq_25948717/article/details/81744277
"""
import random
import operator
def count_total_fruit(data):
    """ 计算各种水果的总数
    return {'banana': 500, 'orange': 300, 'other_fruit': 200}, 1000
    """
    count = {}
    total = 0
    for fruit in data:
        # fruit : banana orange other_fruit, 利用数据的特殊性，求每个类别的样本数
        count[fruit] = data[fruit]['sweet'] + data[fruit]['not_sweet']
        total += count[fruit]

    return count, total


def cal_priori_probs(data):
    """
        计算先验概率 P(X)
    """
    categories, total = count_total_fruit(data)
    prior_probs = {}

    for label in categories:
        prior_prob = categories[label] / total
        prior_probs[label] = prior_prob
    return prior_probs


def likelihood_prob(data):
    """
        各特征值在已知水果下的概率
    """
    count, _ = count_total_fruit(data)
    likelihood = {}

    for fruit in data:
        """创建一个临时字典，存储各个特征值的概率"""
        attr_prob = {}
        for attr in data[fruit]:
            # 计算各个特征值在已知水果下的概率
            # attr : long, not long, sweet, not sweet, yellow, not yellow
            attr_prob[attr] = data[fruit][attr] / count[fruit]
        # 存储每个水果下各个特征的信息
        likelihood[fruit] = attr_prob

    return likelihood


def evidence_prob(data):
    """
        计算特征的概率对分类结果的影响
        return {'long':50%...}
    """
    attrs = list(data['banana'].keys())  # 获得所有水果特征
    count, total = count_total_fruit(data)
    evidence_prob = {}

    # 计算各种特征的概率
    for attr in attrs:
        attr_total = 0
        for fruit in data:
            # 计算所有水果中，相同属性下水果的总量
            attr_total += data[fruit][attr]
        evidence_prob[attr] = attr_total / total
    return evidence_prob


class navie_bayes_classifer:
    """ 初始化朴素贝叶斯， 实例化时调用  P(c|X) = (P(X|c) * P(c)) / P(X)   -> target : 极大后验概率(MAP)
    """
    def __init__(self):
        self._data = datasets
        self._labels = [k for k in self._data.keys()]
        self._priori_prob = cal_priori_probs(self._data)  # 先验概率  P(c)
        self._likelihood_prob = likelihood_prob(self._data)  # 联合概率: 各特征值在已知水果下的概率  P(X|c)
        self._evidence_prob = evidence_prob(self._data)  # P(X)

    def get_label(self, *feature):
        # get_label(self, length, sweetness, color):
        """获得某一组特征值的类别"""
        self._attrs = [*feature]
        # all_attrs = list(self._data[self._labels[0]].keys())   # problem  ['long', 'not_long', 'sweet', 'not_sweet', 'yellow', 'not_yellow']
        result = {}

        for label in self._labels:
            # 取某种水果的占比, 先验概率
            prob = self._priori_prob[label]

            for attr in self._attrs:
                # 单个水果的某个特征概率除以总的某个特征概率 再乘以某水果占比率
                prob *= self._likelihood_prob[label][attr] / self._evidence_prob[attr]
            result[label] = prob

        return result


def random_attr(pair):
    #生成0-1之间的随机数
    return pair[random.randint(0, 1)]


def generate_attrs(test_data_length):
    # 特征值的取值集合
    sets = [('long', 'not_long'), ('sweet', 'not_sweet'), ('yellow', 'not_yellow')]
    test_data = []
    for i in range(test_data_length):
        # 使用map函数来生成一组特征值
        test_data.append(list(map(random_attr, sets)))
    return test_data

def main(test_data_length):
    length = test_data_length
    test_data = generate_attrs(length)
    classifer = navie_bayes_classifer()

    for data in test_data:
        # 预测属于哪种水果的概率
        result = classifer.get_label(*data)
        # 对后验概率排序，输出概率最大的标签
        label = str(sorted(result.items(), key = operator.itemgetter(1), reverse=True)[0][0])

        print(f"特征值： {data}")
        print(f"预测结果：{result}")
        print(f"类别：{label}\n")

if __name__ == '__main__':
    datasets = {
        'banana': {'long': 400, 'not_long': 100, 'sweet': 350, 'not_sweet': 150, 'yellow': 450, 'not_yellow': 50},
        'orange': {'long': 0, 'not_long': 300, 'sweet': 150, 'not_sweet': 150, 'yellow': 300, 'not_yellow': 0},
        'other_fruit': {'long': 100, 'not_long': 100, 'sweet': 150, 'not_sweet': 50, 'yellow': 50, 'not_yellow': 150}
    }

    test_data_length = 20
    main(test_data_length)

