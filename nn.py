from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def main():

    samples: int = 1_000
    X, y = datasets.make_moons(samples, noise=0.2, random_state=42)

    gen: np.Generator = np.random.default_rng(42)

    w0 = gen.random((2, 2))
    w1 = gen.random((2, 2))
    w2 = gen.random(2)
    b0 = gen.random(2)
    b1 = gen.random(2)
    b2 = gen.random(1)

    taxa = 0.01

    _, acc = accuracy(X, y, w0, w1, w2, b0, b1, b2)

    print(acc, "acurácia antes do treinamento")

    max_epochs: int = 10_500

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

    es = EarlyStopping(improvement_rate=0.0001)
    
    for epoch in range(max_epochs):
        epoch_loss: float = 0.

        batches = make_batches(X_train, y_train, gen)

        for X_batch, y_batch in batches:
            batch_size: int = len(X_batch)

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
            for k in range(batch_size):
                g_w0, g_b0, g_w1, g_b1, g_w2, g_b2, L = train(X_batch[k], w0, w1, w2, b0, b1, b2, y_batch[k])
                grad_w0 += g_w0
                grad_w1 += g_w1
                grad_w2 += g_w2
                grad_b0 += g_b0
                grad_b1 += g_b1
                grad_b2 += g_b2
                epoch_loss += L

            w0 -= taxa * grad_w0 / batch_size
            w1 -= taxa * grad_w1 / batch_size
            w2 -= taxa * grad_w2 / batch_size
            b0 -= taxa * grad_b0 / batch_size
            b1 -= taxa * grad_b1 / batch_size
            b2 -= taxa * grad_b2 / batch_size

        test_loss: float = 0.
        S: int = len(X_test)
        for idx in range(S):
            test_loss += loss_calc(
                X_test[idx], w0, w1, w2, b0, b1, b2, y_test[idx]
            )

        if epoch % 100 == 0:
            print(f"epoch {epoch:5d}  train={epoch_loss / len(X_train):.4f}")
        
        if es.step(test_loss, epoch):
            print(f"Treinamento interrompid na época {epoch} devido à constantes faltas de aperfeiçoamento.")
            print(f"Melhor valor de perda registrado: {es.best_loss:.4f} (época {es.best_epoch}).")
            break

    correct_train, acc_train = accuracy(X_train, y_train, w0, w1, w2, b0, b1, b2)

    correct_test, acc_test = accuracy(X_test, y_test, w0, w1, w2, b0, b1, b2)

    print(f"acc treino: {correct_train}/{len(X_train)} = {acc_train:.3f}")
    print(f"acc teste:  {correct_test}/{len(X_test)}   = {acc_test:.3f}")

    train_with_keras(taxa, X_train, y_train)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 18):
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        
        split = int(len(idx) * (1 - test_size))
        
        train_idx.append(idx[:split])
        test_idx.append(idx[split:])

    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def make_batches(X, y, rng, batch_size: int = 16):
    n_samples: int = len(X)
    random_idx = rng.permutation(n_samples)

    batches = []

    for start in range(0, n_samples, batch_size):
        batch_slice = random_idx[start : start + batch_size]
        batches.append((X[batch_slice], y[batch_slice]))

    return batches


def train(x: np.ndarray, w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, d: float):
    """
        O calculo do "processamento" de um neuronio é dado por Y = peso * amostra + vies
    """

    # features passadas para o primeiro neuronio da 1 camada
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y00 = tanh(v0)

    s03 = y00 * w1[0, 0]
    s04 = y00 * w1[0, 1]
    s05 = s03 + s04
    v2 = s05 + b1[0]
    y02 = tanh(v2)

    s06 = y02 * w2[0]
    #### parte de cima do desenho

    # features passadas para o segundo neuronio da 1 camada
    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y10 = tanh(v1)

    s20 = y10 * w1[1, 0]
    s21 = y10 * w1[1, 1]
    s23 = s20 + s21
    v3 = s23 + b1[1]
    y12 = tanh(v3)
    s24 = y12 * w2[1]

    ### parte de baixo

    s30 = s06 + s24
    v4 = s30 + b2[0]

    y30 = sigmoid(v4)

    e = y30 - d

    L = 0.5 * (e ** 2)

    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    grad_L = 1

    grad_e = e * grad_L
    grad_y30 = grad_e
    grad_v4 = grad_y30 * d_sigmoid(y30)
    grad_b2[0] = grad_v4

    # parte de cima do desenho
    grad_y02 = grad_v4 * w2[0]
    grad_w2[0] = grad_v4 * y02

    grad_v2 = grad_y02 * d_tanh(y02)
    grad_b1[0] = grad_v2

    grad_w1[0, 1] = grad_v2 * y00
    grad_y00_b = grad_v2 * w1[0, 1]

    grad_w1[0, 0] = grad_v2 * y00
    grad_y00_a = grad_v2 * w1[0, 0]

    grad_v0 = (grad_y00_a + grad_y00_b) * d_tanh(y00)
    grad_b0[0] = grad_v0

    grad_w0[0, 0] = grad_v0 * x[0]
    grad_w0[0, 1] = grad_v0 * x[1]

    # para de baixo do desenho
    grad_w2[1] = grad_v4 * y12
    grad_y12 = grad_v4 * w2[1]

    grad_v3 = grad_y12 * d_tanh(y12)
    grad_b1[1] = grad_v3

    grad_w1[1, 1] = grad_v3 * y10
    grad_y10_b = grad_v3 * w1[1, 1]

    grad_y10_a = grad_v3 * w1[1, 0]
    grad_w1[1, 0] = grad_v3 * y10

    grad_v1 = (grad_y10_a + grad_y10_b) * d_tanh(y10)

    grad_b0[1] = grad_v1

    grad_w0[1, 0] = grad_v1 * x[0]
    grad_w0[1, 1] = grad_v1 * x[1]
    
    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, L

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(y):
    return 1 - (y ** 2)

def forward(x, w0, w1, w2, b0, b1, b2):
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


def loss_calc(x, w0, w1, w2, b0, b1, b2, d):
    # features passadas para o primeiro neuronio da 1 camada
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y00 = tanh(v0)

    s03 = y00 * w1[0, 0]
    s04 = y00 * w1[0, 1]
    s05 = s03 + s04
    v2 = s05 + b1[0]
    y02 = tanh(v2)

    s06 = y02 * w2[0]
    #### parte de cima do desenho

    # features passadas para o segundo neuronio da 1 camada
    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y10 = tanh(v1)

    s20 = y10 * w1[1, 0]
    s21 = y10 * w1[1, 1]
    s23 = s20 + s21
    v3 = s23 + b1[1]
    y12 = tanh(v3)
    s24 = y12 * w2[1]

    ### parte de baixo

    s30 = s06 + s24
    v4 = s30 + b2[0]

    y30 = sigmoid(v4)

    e = y30 - d

    return 0.5 * (e ** 2)

def train_with_keras(learning_rate: float, X: np.ndarray, Y: np.ndarray, batch_size: int = 16):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, epochs=100, verbose=False, batch_size=batch_size)

    model.evaluate(X, Y)

def accuracy(X: np.ndarray, y: np.ndarray, w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray):
    acc: int = 0
    L: int = len(X)
    for i in range(L):
        predicted = forward(X[i], w0, w1, w2, b0, b1, b2)
        if predicted == y[i]:
            acc += 1

    return acc, acc/L

class EarlyStopping:

    def __init__(self, improvement_rate: float, patience: int = 10) -> None:
        self.improvement_rate = improvement_rate
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch: int = -1

    def step(self, curr_loss: float, curr_epoch: int) -> bool:

        if curr_loss < self.best_loss - self.improvement_rate:
            self.counter = 0
            self.best_loss = curr_loss
            self.best_epoch = curr_epoch
        else:
            self.counter += 1

        return self.counter > self.patience
    
main()