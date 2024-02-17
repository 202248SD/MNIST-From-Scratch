import numpy as np
import pandas as pd


def sigmoid(x, d=False):
    sig = 1 / (1 + np.exp(-x))
    if d:
        return sig * (1 - sig)
    return sig


def ReLU(x, d=False):
    if d:
        return x > 0
    return np.maximum(x, 0)


def softmax(x, d=False):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def MSE(Y, YHat):
    return (np.square(Y - YHat)).mean(axis=0)


data = pd.read_csv('mnist_csv/mnist_train.csv')

data = np.array(data)
m, n = data.shape

data_train = data[0:1024].T
labels = data_train[0]  # 60000
x_train = data_train[1:]  # 784 x 60000
x_train = x_train / 255
print(x_train.shape, labels.shape)


def init_params():
    w1 = np.random.randn(16, n - 1)  # 16 x 784
    b1 = np.random.randn(16, 1)  # 16 x 1
    w2 = np.random.randn(16, 16)  # 16 x 16
    b2 = np.random.rand(16, 1)  # 16 x 1
    w3 = np.random.randn(10, 16)  # 10 x 16
    b3 = np.random.randn(10, 1)  # 10 x 1
    return w1, w2, w3, b1, b2, b3


def feedforward(X, w1, w2, w3, b1, b2, b3):
    z1 = np.dot(w1, X) + b1
    a1 = sigmoid(z1)  # 16 x 60000
    z2 = np.dot(w2, a1) + b2
    a2 = ReLU(z2)  # 16 x 60000
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)  # 10 x 60000
    return a1, a2, a3, z1, z2, z3


def one_hot(Y):
    YHat = np.zeros((Y.size, 10))
    YHat[np.arange(Y.size), Y] = 1
    YHat = YHat.T
    return YHat


def backProp(X, Y, a1, a2, a3, w1, w2, w3, z1, z2, z3, b1, b2, b3):
    Y = one_hot(Y)
    m = X.shape[0]

    dz3 = (a3 - Y) / m
    dw3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=0)

    da2 = np.dot(w3.T, dz3)
    dz2 = da2 * ReLU(z2, d=True)
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=0)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * sigmoid(da1, d=True)
    dw1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=0)

    return dw1, db1, dw2, db2, dw3, db3


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, lr):
    w1 -= lr * dw1
    w2 -= lr * dw2
    w3 -= lr * dw3
    b1 -= lr * np.mean(db1)
    b2 -= lr * np.mean(db2)
    b3 -= lr * np.mean(db3)
    return w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3


def output(a3):
    return np.argmax(a3, axis=0)


def accuracy(output, Y):
    print(output, Y)
    return 100 * np.sum(output == Y) / Y.size


def train_network(X, Y, lr, iterations):
    w1, w2, w3, b1, b2, b3 = init_params()
    for i in range(iterations):
        a1, a2, a3, z1, z2, z3 = feedforward(X, w1, w2, w3, b1, b2, b3)

        cost = MSE(a3, labels)

        dw1, db1, dw2, db2, dw3, db3 = backProp(X, Y, a1, a2, a3, w1, w2, w3, z1, z2, z3, b1, b2, b3)

        w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, lr)

        if i % 100 == 0:
            out = output(a3)
            print("Accuracy:", accuracy(out, Y))
            print(out, Y)

    return w1, w2, w3, b1, b2, b3


w1, w2, w3, b1, b2, b3 = train_network(x_train, labels, 0.25, 4096)









