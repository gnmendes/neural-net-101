from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(y):
    return 1 - (y ** 2)

def simple_feed_foward(x, w0, w1, w2, b0, b1, b2):
    #forward
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y01 = y00 = tanh(v0)

    s03 = y00 * w1[0, 0]
    s04 = y01 * w1[0, 1]
    s05 = s03 + s04

    v2 = s05 + b1[0]
    y02 = tanh(v2)

    s06 = y02 * w2[0]
    #### parte de cima do desenho

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y10 = y11 = tanh(v1)

    s20 = y10 * w1[1, 0]
    s21 = y11 * w1[1, 1]
    s23 = s20 + s21
    v3 = s23 + b1[1]
    y12 = tanh(v3)
    ### parte de baixo

    s24 = y12 * w2[1]
    s30 = s06 + s24
    v4 = s30 + b2[0]

    y30 = sigmoid(v4)

    return 1 if y30 > 0.5 else 0 # relu


def train(x: np.ndarray, w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, d: float):
    """
        O calculo do "processamento" de um neuronio é dado por Y = peso * amostra + vies
    """
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y01 = y00 = tanh(v0)

    s03 = y00 * w1[0, 0]
    s04 = y01 * w1[0, 1]
    s05 = s03 + s04

    v2 = s05 + b1[0]
    y02 = tanh(v2)

    s06 = y02 * w2[0]
    #### parte de cima do desenho

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y10 = y11 = tanh(v1)

    s20 = y10 * w1[1, 0]
    s21 = y11 * w1[1, 1]
    s23 = s20 + s21
    v3 = s23 + b1[1]
    y12 = tanh(v3)
    ### parte de baixo

    s24 = y12 * w2[1]
    s30 = s06 + s24
    v4 = s30 + b2[0]

    y30 = sigmoid(v4)

    e = y30 - d

    L = 1/2 * (e ** 2)

    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    grad_L = 1

    grad_e = e #* grad_L
    grad_y30 = grad_e
    grad_v4 = grad_y30 * d_sigmoid(y30)
    grad_s06 = grad_s24 = grad_s30 = grad_b2[0] = grad_v4

    grad_y02 = grad_s06 * w2[0]
    grad_w2[0] = grad_s06 * y02
    grad_v2 = grad_y02 * d_tanh(y02)
    grad_s03 = grad_s04 = grad_s05 = grad_b1[0] = grad_v2

    grad_w1[0, 1] = grad_s04 * y01
    grad_y01 = grad_s04 * w1[0, 1]

    grad_w1[0, 0] = grad_s03 * y00
    grad_y00 = grad_s03 * w1[0, 0]

    grad_v0 = (grad_y00 + grad_y01) * d_tanh(y00)

    grad_s00 = grad_s01 = grad_s02 = grad_b0[0] = grad_v0

    grad_w0[0, 1] = grad_s01 * x[1]
    grad_w0[0, 0] = grad_s00 * x[0]

    grad_w2[1] = grad_s24 * y12
    grad_y12 = grad_s24 * w2[1]

    grad_s20 = grad_s21 = grad_s23 = grad_b1[1] = grad_v3 = grad_y12 * d_tanh(y12)

    grad_w1[1, 1] = grad_s21 * y11
    grad_y11 = grad_s21 * w1[1, 1]

    grad_y10 = grad_s20 * w1[1, 0]
    grad_w1[1, 0] = grad_s20 * y10

    grad_v1 = (grad_y10 + grad_y11) * d_tanh(y10)

    grad_s10 = grad_s11 = grad_s12 = grad_b0[1] = grad_v1

    grad_w0[1, 0] = grad_s10 * x[0]
    grad_w0[1, 1] = grad_s11 * x[1]
    
    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, L

def main():
    # inicialização aleatória

    X, Y = datasets.make_moons(100, noise=0.2, random_state=42)

    w0 = np.random.rand(2, 2)
    w1 = np.random.rand(2, 2)
    w2 = np.random.rand(2)
    b0 = np.random.rand(2)
    b1 = np.random.rand(2)
    b2 = np.random.rand(1)

    taxa = 0.01

    acc = 0
    for i in range(100):
        out = simple_feed_foward(X[i], w0, w1, w2, b0, b1, b2)
        if out == Y[i]:
            acc += 1

    print(acc, "acurácia antes do treinamento")

    # gradiente descendente
    # epochs

    for i in range(10_000):
        loss = 0

        grad_w0 = np.zeros(w0.shape)
        grad_w1 = np.zeros(w1.shape)
        grad_w2 = np.zeros(w2.shape)
        grad_b0 = np.zeros(b0.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_b2 = np.zeros(b2.shape)

        """
        Batch Gradient Descent. Batch Size = Size of Training Set
            Stochastic Gradient Descent. Batch Size = 1
            Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
        """
        for k in range(100):
            g_w0, g_b0, g_w1, g_b1, g_w2, g_b2, L = train(X[k], w0, w1, w2, b0, b1, b2, Y[k])
            grad_w0 += g_w0
            grad_w1 += g_w1
            grad_w2 += g_w2
            grad_b0 += g_b0
            grad_b1 += g_b1
            grad_b2 += g_b2
            loss += L

        w0 -= taxa * grad_w0
        w1 -= taxa * grad_w1
        w2 -= taxa * grad_w2
        b0 -= taxa * grad_b0
        b1 -= taxa * grad_b1
        b2 -= taxa * grad_b2

        if i % 100 == 0:
            print(i, loss)

    acc = 0
    for i in range(100):
        out = simple_feed_foward(X[i], w0, w1, w2, b0, b1, b2)
        if out == Y[i]:
            acc += 1
    print('acc', acc)

    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=taxa)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, epochs=100, verbose=False, batch_size=5)

    acc = model.evaluate(X, Y)


main()