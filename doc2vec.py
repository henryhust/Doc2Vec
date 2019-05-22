import jieba
import pickle
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def get_data(file_path):
    """
    获取csv当中的评论数据
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)
    data = list(df["evaluation"])
    return data


def cut_word(txt):
    """
    对二维文本数组进行分词切分
    :param txt:输入由文本构成的数组
    :return:返回切分后文本列表
    """
    stoplist = [line.strip() for line in open("./Doc2Vec/中文停用词")]
    lines1 = []
    for line in txt:
        line = line.strip()
        line_cut = jieba.cut(line)
        line_split = " ".join(line_cut).split()
        line_result = [word.strip() for word in line_split if word not in stoplist]
        lines1.append(line_result)
    return lines1


def x_train(data_list):
    """
    构造训练数据
    :param data_list:文本列表
    :return:构造训练数据
    """
    documents = []
    for i, data in enumerate(data_list):
        document = TaggedDocument(data, tags=[i])
        documents.append(document)
    return documents


def train(x_train, size=50):
    """
    模型训练，构建Doc2Vec模型model
    :param x_train:训练数据
    :param size:输出向量维度
    :return:model对象
    """
    model = Doc2Vec(x_train, vector_size=size, window=3, min_count=1, sample=1e-3, nagative=5, workers=4)
    model.train(x_train, total_examples=model.corpus_count, epochs=10)
    return model


def get_vec(model, test_data):
    """
    使用已训练好的模型，对testdata进行预测
    :param model: 已经训练好的模型
    :param test_data: 待预测数据，list类型
    :return: 对应文档向量
    """
    vecs = []
    for data in test_data:
        vec = model.infer_vector(data)
        vecs.append(vec)
    return np.array(vecs)


def o_distance(arr1, arr2):
    """
    计算欧式距离
    :param arr1:
    :param arr2:
    :return:
    """
    distance = np.sqrt(np.sum((arr1-arr2)**2))
    return distance


def save_model(model, file_path):
    """
    保存模型对象
    :param model:已经训练的模型对象
    :param file_path: 文件保存路径
    :return: 打印保存成功消息
    """
    with open(file_path, "wb") as fwb:
        pickle.dump(model, fwb)
    print("模型保存成功, 位置：%s" % (file_path))


if __name__ == '__main__':

    # 训练or测试
    mode = "dev"
    file_path = str("./Doc2Vec/shop_com.csv")
    save_path = "./Doc2Vec/model/doc_emb.pkl"

    # 加载数据
    data = get_data(file_path)

    # 数据分词
    cut_data1 = cut_word(data)
    print(cut_data1)

    # 分词数据格式化
    documents = x_train(cut_data1)

    if mode == "train":
        # 模型训练
        model = train(documents)

        # 模型保存
        save_model(model, save_path)
    else:
        # 模型重新加载
        model = pickle.load(open(save_path, "rb"))


    # 测试词表效果
    str1 = ['正品行货！非常喜欢爱京东']
    str2 = ["是正品！超级喜欢在京东买东西"]
    str3 = ["假货！淘宝东西真的垃圾！"]
    str4 = ["还是喜洋洋受到孩子们的喜爱"]
    vec1 =get_vec(model, cut_word(str1))
    vec2 = get_vec(model, cut_word(str2))
    vec3 = get_vec(model, cut_word(str3))
    vec4 = get_vec(model, cut_word(str4))

    print(o_distance(vec2, vec1))       # 0.30
    print(o_distance(vec2, vec3))       # 0.44
    print(o_distance(vec2, vec4))       # 0.40

