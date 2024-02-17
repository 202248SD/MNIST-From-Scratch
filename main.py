import numpy as np
import pandas as pd
from functions import *

np.seterr(all="ignore")


def init_train_data(size=2048):
    data = pd.read_csv('mnist_csv/mnist_train.csv')
    data = np.array(data)

    data_train = data[0:size].T
    labels = data_train[0]
    x_train = data_train[1:]
    x_train = x_train / 255
    return x_train, labels


def init_params():
    w1 = np.random.randn(16, 784)  # 16 x 784
    b1 = np.random.randn(16, 1)  # 16 x 1
    w2 = np.random.randn(16, 16)  # 16 x 16
    b2 = np.random.rand(16, 1)  # 16 x 1
    w3 = np.random.randn(10, 16)  # 10 x 16
    b3 = np.random.randn(10, 1)  # 10 x 1

    weight_param = [w1, w2, w3]
    bias_param = [b1, b2, b3]

    return weight_param, bias_param


def feedforward(X, weights, biases):
    w1, w2, w3 = weights
    b1, b2, b3 = biases

    z1 = np.dot(w1, X) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = ReLU(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)

    prod_sum = [z1, z2, z3]
    activations = [a1, a2, a3]

    return prod_sum, activations


def one_hot(Y):
    YHat = np.zeros((Y.size, 10))
    YHat[np.arange(Y.size), Y] = 1
    YHat = YHat.T
    return YHat


def backProp(X, Y, weights, biases, prod_sum, activations):
    w1, w2, w3 = weights
    b1, b2, b3 = biases
    z1, z2, z3 = prod_sum
    a1, a2, a3 = activations

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
    gradients = [dw1, dw2, dw3, np.mean(db1), np.mean(db2), np.mean(db3)]

    return gradients


def update_params(weights, biases, gradients, momentum, lr, rho):
    w1, w2, w3 = weights
    b1, b2, b3 = biases
    dw1, dw2, dw3, db1, db2, db3 = gradients
    vw1, vw2, vw3, vb1, vb2, vb3 = momentum

    vw1 = rho * vw1 + lr * dw1
    vw2 = rho * vw2 + lr * dw2
    vw3 = rho * vw3 + lr * dw3
    vb1 = rho * vb1 + lr * db1
    vb2 = rho * vb2 + lr * db2
    vb3 = rho * vb3 + lr * db3

    w1 -= vw1
    w2 -= vw2
    w3 -= vw3
    b1 -= vb1
    b2 -= vb2
    b3 -= vb3

    weights = [w1, w2, w3]
    biases = [b1, b2, b3]
    momentum = [vw1, vw2, vw3, vb1, vb2, vb3]

    return weights, biases, momentum


def output(A):
    *_, a3 = A
    return np.argmax(a3, axis=0)


def accuracy(output, Y):
    return 100 * np.sum(output == Y) / Y.size


def train_network(X, Y, lr, rho, iterations):
    W, B = init_params()
    momentum = [0, 0, 0, 0, 0, 0]
    for i in range(1, iterations):

        Z, A = feedforward(X, W, B)

        gradients = backProp(X, Y, W, B, Z, A)

        W, B, momentum = update_params(W, B, gradients, momentum, lr, rho)

        if i % (iterations // 10) == 0:
            out = output(A)
            print(f"Epoch:{i}, Accuracy: {accuracy(out, Y)}, Cost: {MSE(out, Y)}")
            print(out, Y)
    print(f"Training Accuracy: {accuracy(output(A), Y)}")
    return W, B


def save(weights, biases):
    w1, w2, w3 = weights
    b1, b2, b3 = biases
    np.savez('parameters.npz', w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)


def init_test_data():
    test_data = pd.read_csv('mnist_csv/mnist_test.csv')
    test_data = np.array(test_data).T

    answer = test_data[0]
    test = test_data[1:]
    test = test / 255
    return [test, answer]


def test(X, Y, weights, biases):
    _, A = feedforward(X, weights, biases)
    out = output(A)
    print(f"Test Accuracy: {accuracy(out, Y)}, Cost: {MSE(out, Y)}")
    print(out, Y)
    return accuracy(out, Y)


def load_parameters():
    p = np.load('parameters.npz')
    W = p['w1'], p['w2'], p['w3']
    B = p['b1'], p['b2'], p['b3']
    Z, A = feedforward(x_train, W, B)

    return A, Z, W, B


def summary(A, Y, W, B):
    print(f"Train Accuracy: {accuracy(output(A), Y)}, Cost: {MSE(output(A), Y)}")
    print(output(A), Y)
    test(init_test_data()[0], init_test_data()[1], W, B)


x_train, labels = init_train_data(2048)

# weights, biases = train_network(x_train, labels, 0.1, 0.84, 1000)
# test(init_test_data()[0], init_test_data()[1], weights, biases)

A, _, W, B = load_parameters()
summary(A, labels, W, B)
