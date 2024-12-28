import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def logistic_regression():
    # 1、读取数据
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv(path, names=column_name)

    # 2、缺失值处理
    ## 1）替换-》np.nan
    data = data.replace(to_replace="?", value=np.nan)
    ## 2）删除缺失样本
    data.dropna(inplace=True)
    # print(data.isnull().any()) # 不存在缺失值

    # 2 筛选特征值和目标值
    x = data.iloc[:, 1:-1]
    y = data["Class"]

    # 3、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 4、标准化和归一化处理
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5、预估器流程
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 逻辑回归的模型参数：回归系数和偏置
    print("回归系数是：\n", estimator.coef_)
    print("偏置是：\n", estimator.intercept_)

    # 6、模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 方法3：查看精确率、召回率、F1-score
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
    print("精确率、召回率、F1-score为：\n",report)

    # 方法4：ROC-AUC
    # y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
    # 将y_test 转换成 0 1
    y_true = np.where(y_test > 3, 1, 0)
    print("ROC为：\n",roc_auc_score(y_true, y_predict))

    return None

if __name__ == "__main__":
    logistic_regression()