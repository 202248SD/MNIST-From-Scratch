import mnist_csv
import numpy as np
import pandas as pd
import matplotlib as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


data = pd.read_csv('mnist_csv/mnist_train.csv')

data = np.array(data)
m, n = data.shape

data_train = data.T
labels = data_train[0]  # 60000
x_train = data_train[1:]  # 784 x 60000
x_train = x_train / 255


def init_params():
    w1 = np.random.randn(16, n - 1)  # 16 x 784
    b1 = np.random.randn(16, 1)  # 16 x 1
    w2 = np.random.randn(16, 16)  # 16 x 16
    b2 = np.random.rand(16, 1)  # 16 x 1
    w3 = np.random.randn(10, 16)  # 10 x 16
    b3 = np.random.randn(10, 1)  # 10 x 1
    return w1, b1, w2, b2, w3, b3


def feedforward(a0, w1, b1, w2, b2, w3, b3):
    a1 = sigmoid(np.dot(w1, a0) + b1)  # 16 x 60000
    a2 = ReLU(np.dot(w2, a1) + b2)  # 16 x 60000
    a3 = softmax(np.dot(w3, a2) + b3)  # 10 x 60000
    return a1, a2, a3


a1, a2, a3 = feedforward(x_train, *init_params())



