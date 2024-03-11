import numpy as np
import torch

class DataReader:
    def __init__(self, feature_path, label_path):
        """
        初始化DataReader类。

        参数:
        feature_path: 特征数据文件的路径。
        label_path: 标签数据文件的路径。
        """
        self.feature_path = feature_path
        self.label_path = label_path

    def read_data(self):
        """
        读取特征数据和标签数据。

        返回:
        X: 特征数据集，一个numpy数组。
        Y: 标签集，一个numpy数组。
        """
        # 读取特征数据
        with open(self.feature_path, 'r') as f:
            X = [list(map(float, line.split(" "))) for line in f.readlines()]
            X = [np.array(each).reshape([-1, 26]) for each in X]

        # 读取标签数据
        with open(self.label_path, 'r') as f:
            Y = [list(map(int, line.strip())) for line in f.readlines()]
            Y = [np.array(each) for each in Y]

        return X, Y

if __name__ == '__main__':
    # 假设特征数据和标签数据的文件路径分别是'features.txt'和'labels.txt'
    feature_path = '../data/train_add_norm.dat'
    label_path = '../data/train_label.dat'

    # 创建DataReader实例
    data_reader = DataReader(feature_path, label_path)

    # 读取数据
    X, Y = data_reader.read_data()
    # 打印结果以验证
    print("特征数据X的形状:", [each.shape for each in X])
    print("标签数据Y的形状:", [each.shape for each in Y])
