from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def random_forest_titanic():
    """
    用决随机森林对坦坦尼克号乘客的生存进行预测
    :return:
    """
    # 1）获取数据集
    titanic = pd.read_csv("titanic.csv")

    # 2） 数据处理
    ## 特征值和目标值提取
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]
    # ## 缺失值处理
    x["pclass"] = x["pclass"].map({"1st": 1, "2nd": 2, "3rd": 3,})
    x["age"] = x["age"].fillna(x["age"].mean())
    x["sex"] = x["sex"].map({"male": 1, "female": 0})

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3）随机森林预估器
    # estimator = RandomForestClassifier()

    # 4) 网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200],
                  "max_depth": [2, 8, 10, 15, 30, 50]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)

    # # 5）模型评估
    # # 方法1：直接比对真实值和预测值
    # y_predict = estimator.predict(x_test)
    # print("y_predict:\n", y_predict)
    # print("直接比对真实值和预测值:\n", y_test == y_predict)
    #
    # # 方法2：计算准确率
    # score = estimator.score(x_test, y_test)
    # print("准确率为：\n", score)
    #
    # # 最佳参数：best_params_
    # print("最佳参数：\n", estimator.best_params_)
    # # 最佳结果：best_score_
    # print("最佳结果：\n", estimator.best_score_)
    # # 最佳估计器：best_estimator_
    # print("最佳估计器:\n", estimator.best_estimator_)
    # # 交叉验证结果：cv_results_
    # print("交叉验证结果:\n", estimator.cv_results_)

    return None

if __name__ == "__main__":
    random_forest_titanic()