import numpy as np
import matplotlib.pyplot as plt

# Classe MLP (sem mudanças)
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, Y, output):
        m = X.shape[0]
        self.output_error = Y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        self.z1_error = self.output_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * self.sigmoid_derivative(self.a1)
        self.W2 += self.a1.T.dot(self.output_delta) / m
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) / m
        self.W1 += X.T.dot(self.z1_delta) / m
        self.b1 += np.sum(self.z1_delta, axis=0, keepdims=True) / m
    
    def train(self, X, Y, epochs, learning_rate):
        self.learning_rate = learning_rate
        errors = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output)
            error = np.mean(np.square(Y - output))
            errors.append(error)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Error: {error}')
        
        return errors

# Carregar os dados
X = np.load('X.npy')
Y = np.load('Y_classe.npy')

# Verificar as formas dos dados
print(f'Original X shape: {X.shape}')
print(f'Original Y shape: {Y.shape}')

# Ajustar a forma de X se necessário
if X.ndim == 4 and X.shape[-1] == 1:
    X = X.reshape(X.shape[0], -1)

# Verificar a nova forma de X
print(f'Adjusted X shape: {X.shape}')

# Definir os tamanhos das camadas
input_size = X.shape[1]
hidden_size = 10
output_size = Y.shape[1]

# Inicializar a rede
mlp = MLP(input_size, hidden_size, output_size)

# Treinar a rede
epochs = 1000
learning_rate = 0.1
errors = mlp.train(X, Y, epochs, learning_rate)

# Plotar e salvar os erros
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Error')
plt.savefig('training_error.png')  # Salvar o gráfico como PNG
plt.show()
