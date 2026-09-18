from IPython.display import clear_output
import matplotlib.pyplot as plt

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def draw_step(active_node: str, step_name: str, values: dict, weights: dict, grads: dict = None, delay: float = 2.0):
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')
    
    layer_x = [1, 3.5, 6, 8.5, 11]
    nodes_pos = {
        'x1': (layer_x[0], 4.0), 'x2': (layer_x[0], 2.0),
        'v0': (layer_x[1], 4.5), 'v1': (layer_x[1], 1.5),
        'v2': (layer_x[2], 4.5), 'v3': (layer_x[2], 1.5),
        'y30': (layer_x[3], 3.0), 'loss': (layer_x[4], 3.0)
    }

    edges_map = {
        ('x1', 'v0'): 'w0_00', ('x2', 'v0'): 'w0_01',
        ('x1', 'v1'): 'w0_10', ('x2', 'v1'): 'w0_11',
        ('v0', 'v2'): 'w1_00', ('v0', 'v3'): 'w1_01',
        ('v1', 'v2'): 'w1_10', ('v1', 'v3'): 'w1_11',
        ('v2', 'y30'): 'w2_0',  ('v3', 'y30'): 'w2_1',
        ('y30', 'loss'): None
    }

    is_backprop = 'Backprop' in step_name

    for (start, end), w_key in edges_map.items():
        x1, y1 = nodes_pos[start]
        x2, y2 = nodes_pos[end]
        is_active = (start == active_node or end == active_node)
        
        color = '#e74c3c' if (is_backprop and is_active) else ('#2980b9' if is_active else '#bdc3c7')
        lw = 2.5 if is_active else 1.0
        
        ax.annotate('', xy=(x2-0.5, y2), xytext=(x1+0.5, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10))

        if w_key and w_key in weights:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.15, f"{weights[w_key]:.2f}", fontsize=7.5, color='#c0392b',
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    for node, (nx, ny) in nodes_pos.items():
        is_current = (node == active_node)
        color = '#e74c3c' if (is_backprop and is_current) else ('#f1c40f' if is_current else '#34495e')
        edge_color = '#c0392b' if (is_backprop and is_current) else ('#e67e22' if is_current else 'black')
        
        # Exibe o valor do gradiente se estiver no Backprop, ou a ativação no Forward
        if is_backprop and grads and node in grads:
            label_text = f"{node}\ngrad:\n{grads[node]:.3f}"
        else:
            val = values.get(node)
            label_text = f"{node}\n{val:.2f}" if val is not None else node

        circle = plt.Circle((nx, ny), 0.55, color=color, ec=edge_color, lw=2 if is_current else 1, zorder=4)
        ax.add_patch(circle)
        ax.text(nx, ny, label_text, ha='center', va='center', color='white' if not is_current else 'black', fontweight='bold', fontsize=7.5, zorder=5)

    ax.set_title(f"Passo Atual: {step_name}", fontsize=11, fontweight='bold', pad=15)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    plt.show()
    plt.pause(delay)

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


def train(x: np.ndarray, w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, d: float, debug_plot: bool = False):
    """
        O calculo do "processamento" de um neuronio é dado por Y = peso * amostra + vies
    """
    # Extrai a estrutura atual dos pesos para exibição
    w_dict = {
        'w0_00': w0[0, 0], 'w0_01': w0[0, 1],
        'w0_10': w0[1, 0], 'w0_11': w0[1, 1],
        'w1_00': w1[0, 0], 'w1_01': w1[0, 1],
        'w1_10': w1[1, 0], 'w1_11': w1[1, 1],
        'w2_0': w2[0],     'w2_1': w2[1]
    }

    vals = {'x1': x[0], 'x2': x[1]}
    
    # 1. Entrada
    # if debug_plot: draw_step('x1', '1. Recebendo Entradas (X)', vals)
    if debug_plot: draw_step('x1', '1. Recebendo Entradas', vals, w_dict)

    # 2. Neurônio v0 (Topo Camada 1)
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y01 = y00 = tanh(v0)
    vals['v0'] = y00
    # if debug_plot: draw_step('v0', f'Calculado v0 = tanh({v0:.2f}) -> y00 = {y00:.2f}', vals)
    if debug_plot: draw_step('v0', f'2. v0 = tanh({v0:.2f}) -> {y00:.2f}', vals, w_dict)

    s03 = y00 * w1[0, 0]
    s04 = y01 * w1[0, 1]
    s05 = s03 + s04

    v2 = s05 + b1[0]
    y02 = tanh(v2)
    vals['v2'] = y02
    # if debug_plot: draw_step('v2', f'Calculado v2 = tanh({v2:.2f}) -> y02 = {y02:.2f}', vals)
    if debug_plot: draw_step('v2', f'3. v2 = tanh({v2:.2f}) -> {y02:.2f}', vals, w_dict)

    s06 = y02 * w2[0]
    #### parte de cima do desenho

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y10 = y11 = tanh(v1)
    vals['v1'] = y10
    # if debug_plot: draw_step('v1', f'Calculado v1 = tanh({v1:.2f}) -> y10 = {y10:.2f}', vals)
    if debug_plot: draw_step('v1', f'4. v1 = tanh({v1:.2f}) -> {y10:.2f}', vals, w_dict)
    

    s20 = y10 * w1[1, 0]
    s21 = y11 * w1[1, 1]
    s23 = s20 + s21
    v3 = s23 + b1[1]
    y12 = tanh(v3)
    vals['v3'] = y12
    # if debug_plot: draw_step('v3', f'Calculado v3 = tanh({v3:.2f}) -> y12 = {y12:.2f}', vals)
    if debug_plot: draw_step('v3', f'5. v3 = tanh({v3:.2f}) -> {y12:.2f}', vals, w_dict)
    ### parte de baixo

    s24 = y12 * w2[1]
    s30 = s06 + s24
    v4 = s30 + b2[0]

    y30 = sigmoid(v4)
    vals['y30'] = y30
    # if debug_plot: draw_step('y30', f'Calculada Saída y30 = sigmoid({v4:.2f}) -> {y30:.2f}', vals)
    if debug_plot: draw_step('y30', f'6. Saída y30 = {y30:.2f}', vals, w_dict)

    e = y30 - d

    L = 1/2 * (e ** 2)
    vals['loss'] = L
    # if debug_plot: draw_step('loss', f'Calculada Perda MSE L = {L:.4f}', vals)
    if debug_plot: draw_step('loss', f'7. Perda MSE L = {L:.4f}', vals, w_dict)

    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    grads = {}

    grad_L = 1

    grad_e = e #* grad_L
    grad_y30 = grad_e
    grad_v4 = grad_y30 * d_sigmoid(y30)
    grad_s06 = grad_s24 = grad_s30 = grad_b2[0] = grad_v4
    grads['y30'] = grad_v4
    if debug_plot: draw_step('y30', f'8. [Backprop] Gradiente Saída v4 = {grad_v4:.4f}', vals, w_dict, grads)
    

    grad_y02 = grad_s06 * w2[0]
    grad_w2[0] = grad_s06 * y02
    grad_v2 = grad_y02 * d_tanh(y02)
    grad_s03 = grad_s04 = grad_s05 = grad_b1[0] = grad_v2
    grads['v2'] = grad_v2
    if debug_plot: draw_step('v2', f'9. [Backprop] Gradiente v2 = {grad_v2:.4f}', vals, w_dict, grads)

    grad_w1[0, 1] = grad_s04 * y01
    grad_y01 = grad_s04 * w1[0, 1]

    grad_w1[0, 0] = grad_s03 * y00
    grad_y00 = grad_s03 * w1[0, 0]

    grad_v0 = (grad_y00 + grad_y01) * d_tanh(y00)

    grads['v0'] = grad_v0
    if debug_plot: draw_step('v0', f'10. [Backprop] Gradiente v0 = {grad_v0:.4f}', vals, w_dict, grads)

    grad_s00 = grad_s01 = grad_s02 = grad_b0[0] = grad_v0

    grad_w0[0, 1] = grad_s01 * x[1]
    grad_w0[0, 0] = grad_s00 * x[0]

    grad_w2[1] = grad_s24 * y12
    grad_y12 = grad_s24 * w2[1]

    grad_s20 = grad_s21 = grad_s23 = grad_b1[1] = grad_v3 = grad_y12 * d_tanh(y12)
    grads['v3'] = grad_v3
    if debug_plot: draw_step('v3', f'11. [Backprop] Gradiente v3 = {grad_v3:.4f}', vals, w_dict, grads)

    grad_w1[1, 1] = grad_s21 * y11
    grad_y11 = grad_s21 * w1[1, 1]

    grad_y10 = grad_s20 * w1[1, 0]
    grad_w1[1, 0] = grad_s20 * y10

    grad_v1 = (grad_y10 + grad_y11) * d_tanh(y10)
    grads['v1'] = grad_v1
    if debug_plot: draw_step('v1', f'12. [Backprop] Gradiente v1 = {grad_v1:.4f}', vals, w_dict, grads)

    grad_s10 = grad_s11 = grad_s12 = grad_b0[1] = grad_v1

    grad_w0[1, 0] = grad_s10 * x[0]
    grad_w0[1, 1] = grad_s11 * x[1]
    
    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, L

def main():
    # inicialização aleatória

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

    acc = 0
    for i in range(100):
        out = forward(X[i], w0, w1, w2, b0, b1, b2)
        if out == y[i]:
            acc += 1

    print(acc, "acurácia antes do treinamento")

    max_epochs: int = 10_000

    print("Iniciando animação passo a passo para a primeira amostra...")
    
    # Executa a visualização interativa apenas para a primeira amostra
    train(X[0], w0, w1, w2, b0, b1, b2, y[0], debug_plot=True)
    
    # Desativa o modo interativo e fecha a janela para liberar o Python
    plt.ioff()
    plt.close()
    
    print("Animação concluída. Iniciando o treinamento pesado...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

    losses = []

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

        # Armazena a perda média por amostra na época
        losses.append(epoch_loss / len(X_train))

        if epoch % 100 == 0:
            print(f"epoch {epoch:5d}  train={epoch_loss:.4f}")

    acc_train = 0
    n_train: int = len(X_train)

    for i in range(n_train):
        out = forward(X_train[i], w0, w1, w2, b0, b1, b2)
        if out == y_train[i]:
            acc_train += 1

    acc_test: int = 0
    n_test: int = len(X_test)
    for i in range(n_test):
        out = forward(X_test[i], w0, w1, w2, b0, b1, b2)
        if out == y_test[i]:
            acc_test += 1


    print(f"\nacc treino: {acc_train}/{n_train} = {acc_train/n_train:.3f}")
    print(f"acc val:    {acc_test}/{n_test} = {acc_test/n_test:.3f}")

    train_with_keras(taxa, X_train, y_train, batch_size=5)


def make_batches(X, y, rng, batch_size: int = 16):
    n_samples: int = len(X)
    random_idx = rng.permutation(n_samples)

    batches = []

    for start in range(0, n_samples, batch_size):
        batch_slice = random_idx[start : start + batch_size]
        batches.append((X[batch_slice], y[batch_slice]))

    return batches

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 18):
    num_samples: int = len(X)
    gen = np.random.default_rng(random_state)

    indexes = gen.permutation(num_samples)
    test_samples = int(num_samples * test_size)

    train_indexes = indexes[test_samples:]
    test_indexes = indexes[:test_samples]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]

def train_with_keras(learning_rate: float, X: np.ndarray, Y: np.ndarray, batch_size: int = 16):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, epochs=100, verbose=False, batch_size=batch_size)

    model.evaluate(X, Y)

main()
