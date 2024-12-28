from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd



def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3] # 取所有行，前三列
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer1 = MinMaxScaler() # 默认0-1
    transfer2 = MinMaxScaler(feature_range=(2, 3))

    # 3、调用fit_transform
    data_new1 = transfer1.fit_transform(data)
    data_new2 = transfer2.fit_transform(data)
    print("data_new1:\n", data_new1)
    print("data_new2:\n", data_new2)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2、实例化一个转换器类
    transfer = StandardScaler()

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    return None

if __name__ == "__main__":
    # 代码1：归一化
    # minmax_demo()
    # 代码2：标准化
    # stand_demo()
    print("我爱中国")