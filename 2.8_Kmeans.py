import pandas as pd
from sklearn.cluster import KMeans

def Kmeans():
    # 1、获取数据
    order_products = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")

    # 2、合并表
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])

    # 3、找到user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])
    data_new = table[:10000]

    # 4. 构建模型
    estimator = KMeans(n_clusters=3)
    estimator.fit(data_new)
    y_predict = estimator.predict(data_new)
    print(y_predict[:300])

    return None

if __name__ == "__main__":
    Kmeans()
