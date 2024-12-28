from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=True)

    # 2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(), type(data_new))
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def count_demo():
    """
    文本特征抽取：CountVecotrizer,统计每个样本词出现的个数
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])

    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    # print("data_new:\n", data_new) # 输出为spares矩阵
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def count_chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data1 = ["我爱北京天安门", "天安门上太阳升"]
    data2 = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # 1、实例化一个转换器类
    transfer1 = CountVectorizer()
    transfer2 = CountVectorizer()

    # 2、调用fit_transform
    data1_new = transfer1.fit_transform(data1)
    data2_new = transfer2.fit_transform(data2)
    print("data1_new:\n", data1_new.toarray())
    print("data2_new:\n", data2_new.toarray())
    print("特征名字：\n", transfer1.get_feature_names_out())
    print("特征名字：\n", transfer2.get_feature_names_out())

    return None

# 上面自动分词太傻比了

def cut_word(text):
    """
    进行中文自动分词："我爱北京天安门" --> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))

def count_chinese_demo2():
    """
    中文文本特征抽取，自动分词
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取,找出重要的词汇
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

if __name__ == "__main__":
    # 代码1：字典特征抽取
    # dict_demo()
    # 代码2：文本特征抽取：CountVecotrizer
    # count_demo()
    # 代码3：中文特征抽取
    # count_chinese_demo()
    # 代码4：自动分词
    # print(cut_word("我爱北京天安门"))
    # print(cut_word("北大清华是中国最好的大学"))
    # 代码5：中文文本特征抽取，自动分词
    # count_chinese_demo2()
    # 代码6：用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()
    print("我爱中国")