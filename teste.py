import numpy as np

# Load the data
X = np.load('X.npy')
y = np.load('Y_classe.npy')

# Reshape X to a 2D array
X = X.reshape(-1, 120)

# Define the number of input features, hidden units, and output classes
n_features = X.shape[1]
n_hidden = 100
n_classes = 1

# Initialize the weights and biases for the MLP
np.random.seed(1)
w1 = np.random.randn(n_features, n_hidden)
b1 = np.random.randn(n_hidden)
w2 = np.random.randn(n_hidden, n_classes)
b2 = np.random.randn(n_classes)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the forward pass of the MLP
def forward(X):
    z1 = X.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    y_pred = sigmoid(z2)
    return y_pred

# Define the loss function for the MLP
def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Define the derivative of the loss function with respect to the output of the MLP
def loss_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_pred * (1 - y_pred)

# Define the number of training epochs and the learning rate
epochs = 100
learning_rate = 0.1

# Train the MLP using gradient descent
for e in range(epochs):
    y_pred = forward(X)
    loss_val = loss(y, y_pred)
    gradient_w1 = X.T.dot(loss_derivative(y, y_pred) * sigmoid_derivative(z1))
    gradient_b1 = np.sum(loss_derivative(y, y_pred) * sigmoid_derivative(z1), axis=0)
    gradient_w2 = a1.T.dot(loss_derivative(y, y_pred))
    gradient_b2 = np.sum(loss_derivative(y, y_pred), axis=0)
    w1 -= learning_rate * gradient_w1
    b1 -= learning_rate * gradient_b1
    w2 -= learning_rate * gradient_w2
    b2 -= learning_rate * gradient_b2
    if e % 10 == 0:
        print('Epoch %d, Loss: %.3f' % (e, loss_val))

# Use the trained MLP to make predictions on new data
X_new = np.random.rand(1, n_features)
y_new = forward(X_new)
print('Predicted class:', np.argmax(y_new))