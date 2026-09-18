import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


DEFAULT_EPOCHS = 10_000
DEFAULT_NEURONS = 2
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_TOTAL_LAYERS = 3  # duas camadas ocultas + camada de saída
DEFAULT_ACTIVATION = "tanh"
ANALYSIS_DIR = Path(__file__).resolve().parent / "analises"
_animation_figure = None

def draw_step(active_node: str, step_name: str, values: dict, weights: dict, grads: dict = None, delay: float = 2.0):
    global _animation_figure
    if _animation_figure is None or not plt.fignum_exists(_animation_figure.number):
        _animation_figure, ax = plt.subplots(figsize=(11, 5.5))
        plt.show(block=False)
    else:
        ax = _animation_figure.axes[0]
        ax.clear()

    fig = _animation_figure
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
    # Mantém uma única janela e processa os eventos do backend do macOS.
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
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

def _activation(name):
    """Retorna a ativação e sua derivada (a derivada recebe a saída)."""
    if name == "tanh":
        return np.tanh, lambda value: 1 - value ** 2
    if name == "sigmoid":
        def sigmoid_stable(value):
            value = np.clip(value, -500, 500)
            return 1 / (1 + np.exp(-value))
        return sigmoid_stable, lambda value: value * (1 - value)
    if name == "relu":
        return lambda value: np.maximum(0, value), lambda value: (value > 0).astype(float)
    raise ValueError("A ativação deve ser tanh, sigmoid ou relu.")


def _prompt_int(label, default, minimum=1):
    while True:
        answer = input(f"{label} [padrão: {default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Informe um número inteiro maior ou igual a {minimum}.")


def get_config(args):
    if args.non_interactive:
        epochs, neurons = args.epochs, args.neurons
        hidden_layers, total_layers = args.hidden_layers, args.layers
        activation = args.activation
    else:
        print("\nParâmetros do experimento (pressione Enter para usar o padrão):")
        epochs = _prompt_int("Número de épocas", DEFAULT_EPOCHS)
        neurons = _prompt_int("Número de neurônios em cada camada oculta", DEFAULT_NEURONS)
        hidden_layers = _prompt_int("Número de camadas ocultas", DEFAULT_HIDDEN_LAYERS)
        total_layers = _prompt_int(
            "Número total de camadas (ocultas + saída)", DEFAULT_TOTAL_LAYERS, 2
        )
        activation = input(
            f"Função de ativação das camadas ocultas (tanh/sigmoid/relu) "
            f"[padrão: {DEFAULT_ACTIVATION}]: "
        ).strip().lower() or DEFAULT_ACTIVATION

    while total_layers != hidden_layers + 1:
        message = (
            "O número total de camadas deve ser igual ao número de camadas "
            "ocultas + 1 (camada de saída)."
        )
        if args.non_interactive:
            raise ValueError(message)
        print(message)
        total_layers = _prompt_int(
            "Número total de camadas (ocultas + saída)", hidden_layers + 1, 2
        )
    _activation(activation)
    return epochs, neurons, hidden_layers, total_layers, activation


def train_network(X, y, epochs, neurons, hidden_layers, activation_name, learning_rate=0.01):
    """Treina uma rede densa configurável e retorna pesos, perdas e acurácias."""
    rng = np.random.default_rng(42)
    activation, derivative = _activation(activation_name)
    sizes = [X.shape[1]] + [neurons] * hidden_layers + [1]
    weights = [rng.normal(0, 0.5, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
    biases = [np.zeros(size) for size in sizes[1:]]
    losses, accuracies = [], []

    for epoch in range(epochs):
        indexes = rng.permutation(len(X))
        epoch_loss = 0.0
        for index in indexes:
            activations = [X[index]]
            for layer, (weight, bias) in enumerate(zip(weights, biases)):
                z = activations[-1] @ weight + bias
                activations.append(1 / (1 + np.exp(-np.clip(z, -500, 500))) if layer == len(weights) - 1 else activation(z))

            error = activations[-1][0] - y[index]
            epoch_loss += 0.5 * error ** 2
            delta = np.array([error * activations[-1][0] * (1 - activations[-1][0])])
            grad_weights = [None] * len(weights)
            grad_biases = [None] * len(biases)
            for layer in range(len(weights) - 1, -1, -1):
                grad_weights[layer] = np.outer(activations[layer], delta)
                grad_biases[layer] = delta
                if layer:
                    delta = (weights[layer] @ delta) * derivative(activations[layer])
            for layer in range(len(weights)):
                weights[layer] -= learning_rate * grad_weights[layer]
                biases[layer] -= learning_rate * grad_biases[layer]

        predictions = predict_network(X, weights, biases, activation_name)
        losses.append(epoch_loss / len(X))
        accuracies.append(np.mean(predictions == y))
        if epoch % max(1, epochs // 10) == 0:
            print(f"época {epoch + 1:5d}/{epochs}  perda={losses[-1]:.4f}  acurácia={accuracies[-1]:.3f}")
    return weights, biases, losses, accuracies


def predict_network(X, weights, biases, activation_name):
    activation, _ = _activation(activation_name)
    values = X
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        values = values @ weight + bias
        values = 1 / (1 + np.exp(-np.clip(values, -500, 500))) if layer == len(weights) - 1 else activation(values)
    return (values.ravel() >= 0.5).astype(int)


def save_plots(losses, accuracies):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(losses) + 1)
    loss_path = ANALYSIS_DIR / "neural_net_v2_funcao_perda.png"
    metrics_path = ANALYSIS_DIR / "neural_net_v2_acuracia.png"
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, losses, color="#c0392b")
    plt.title("Função de perda por época")
    plt.xlabel("Época")
    plt.ylabel("Perda MSE")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, accuracies, color="#2980b9")
    plt.title("Acurácia de treinamento por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"Gráficos salvos em: {loss_path} e {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Experimento didático de uma rede neural")
    parser.add_argument("--non-interactive", action="store_true", help="usa os valores dos argumentos")
    parser.add_argument("--skip-animation", action="store_true", help="não abre a animação passo a passo")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--neurons", type=int, default=DEFAULT_NEURONS)
    parser.add_argument("--hidden-layers", type=int, default=DEFAULT_HIDDEN_LAYERS)
    parser.add_argument("--layers", type=int, default=DEFAULT_TOTAL_LAYERS)
    parser.add_argument("--activation", choices=("tanh", "sigmoid", "relu"), default=DEFAULT_ACTIVATION)
    args = parser.parse_args()
    epochs, neurons, hidden_layers, _, activation = get_config(args)

    X, y = datasets.make_moons(1_000, noise=0.2, random_state=42)
    if not args.skip_animation:
        print("Iniciando animação passo a passo para a primeira amostra...")
        plt.ion()
        gen = np.random.default_rng(42)
        animation_weights = [gen.random((2, 2)), gen.random((2, 2)), gen.random(2)]
        animation_biases = [gen.random(2), gen.random(2), gen.random(1)]
        train(X[0], *animation_weights, *animation_biases, y[0], debug_plot=True)
        plt.ioff()
        plt.close("all")
        _animation_figure = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)
    weights, biases, losses, accuracies = train_network(
        X_train, y_train, epochs, neurons, hidden_layers, activation
    )
    train_accuracy = np.mean(predict_network(X_train, weights, biases, activation) == y_train)
    test_accuracy = np.mean(predict_network(X_test, weights, biases, activation) == y_test)
    print(f"\nAcurácia treino: {train_accuracy:.3f}")
    print(f"Acurácia teste:  {test_accuracy:.3f}")
    save_plots(losses, accuracies)


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
