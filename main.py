import numpy as np

def print_result(matrix):
    # Convertendo cada valor para decimal e armazenando em uma nova matriz
    decimal_values = []
    for fila in matrix:
        for value in fila:
           # Especificando 4 casas decimais (ajuste de acordo com sua necessidade)
            decimal_value = f"{value:.4f}"
            decimal_values.append(decimal_value)

    # Transformando a lista em uma matriz NumPy de 1 coluna
    decimal_values = np.array(decimal_values).reshape(-1, 1)

    print(decimal_values)

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Carregar dados do arquivo CSV
data = np.genfromtxt('caracteres-limpos.csv', delimiter=',')

# Extrair os dados de entrada e saída esperada
X = data[:, :-1]
y = data[:, -1].reshape(-1, 1)  # reshape para garantir o formato correto da matriz


# Definir hiperparâmetros
epochs = 1000000  # Número de épocas de treinamento
lr = 0.1  # Taxa de aprendizado


# Inicialização aleatória dos pesos e biases
np.random.seed(1)
input_neurons = X.shape[1]  # Número de features (colunas) nos dados de entrada
hidden_neurons = 10
output_neurons = 1

weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Treinamento da rede neural
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Backpropagation

    # Calculando o erro
    error = y - output
    
    # Calculando os gradientes da camada de saída
    slope_output_layer = sigmoid_derivative(output)
    delta_output = error * slope_output_layer

    # Calculando os gradientes da camada oculta
    error_hidden_layer = delta_output.dot(weights_hidden_output.T)
    slope_hidden_layer = sigmoid_derivative(hidden_layer_output)
    delta_hidden = error_hidden_layer * slope_hidden_layer

    # Atualizando os pesos e biases
    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * lr
    weights_input_hidden += X.T.dot(delta_hidden) * lr
    bias_output += np.sum(delta_output, axis=0) * lr
    bias_hidden += np.sum(delta_hidden, axis=0) * lr

# Testando a rede neural treinada
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output = sigmoid(output_layer_input)


print("Saída prevista:")
print_result(output)