from sklearn.utils import all_estimators

def all_learner():
    """
    所有
    :return:
    """
    for name, class_ in all_estimators():
        print(name, ":", class_)
    return

def classifier():
    """
    分类模型，用于处理分类任务（目标变量是离散的类别标签）
    :return:
    """
    for name, class_ in all_estimators(type_filter="classifier"):
        print(name, ":", class_)
    return

def regressor():
    """
    回归模型，用于处理回归任务（目标变量是连续值）
    :return:
    """
    for name, class_ in all_estimators(type_filter="regressor"):
        print(name, ":", class_)
    return

def cluster():
    """
    聚类算法，用于无监督学习将样本分组
    :return:
    """
    for name, class_ in all_estimators(type_filter="cluster"):
        print(name, ":", class_)
    return

def transformer():
    """
    变换器，用于数据的特征工程、降维、标准化等预处理
    :return:
    """
    for name, class_ in all_estimators(type_filter="transformer"):
        print(name, ":", class_)
    return

if __name__ == "__main__":
    # all_learner()
    # classifier()
    # regressor()
    cluster()
    # transformer()