from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# from scipy.stats import pearsonr
# import jieba
# import pandas as pd


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    # print("鸢尾花数据集：\n", iris)
    # print("查看数据集描述：\n", iris["DESCR"])
    # print("查看特征值的名字：\n", iris.feature_names)
    # print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分
    # x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    # print("训练集的特征值：\n", x_train, x_train.shape)

    return None

datasets_demo()