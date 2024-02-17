import numpy as np


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