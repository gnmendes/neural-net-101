# Neural Net 101

Experimento didático de uma rede neural para classificação do conjunto `make_moons`, com animação do forward/backpropagation, treinamento manual e geração de gráficos.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução

Execução interativa com os valores padrão:

```bash
python neural_net_v2.py
```

Execução sem prompts e sem animação:

```bash
python neural_net_v2.py --non-interactive --skip-animation
```

Os parâmetros padrão são 10.000 épocas, 2 neurônios em cada camada oculta, 2 camadas ocultas, 3 camadas totais e ativação `tanh`. As ativações disponíveis são `tanh`, `sigmoid` e `relu`.

## Resultados

Os gráficos e o relatório comparativo ficam em [`analises/`](analises/). O arquivo [`relatorio_comparativo_neural_net_v2.md`](analises/relatorio_comparativo_neural_net_v2.md) documenta as quatro rodadas realizadas e seus resultados. A melhor configuração observada foi a quarta: 6.000 épocas, 4 neurônios por camada, 4 camadas ocultas, 5 camadas totais e `tanh`.
