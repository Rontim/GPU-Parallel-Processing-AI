import numpy as np
import cupy as cp
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from .utils import to_device, to_cpu, sigmoid, tanh, binary_cross_entropy, has_cupy


class NeuralNetwork:
    def __init__(self, n_input=2, n_hidden=16, n_output=1, use_gpu=False):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.use_gpu = use_gpu and has_cupy()

        # Initialize weights and biases
        if self.use_gpu:
            self.W1 = cp.random.randn(
                n_input, n_hidden) * cp.sqrt(2. / n_input)
            self.b1 = cp.zeros((1, n_hidden))
            self.W2 = cp.random.randn(
                n_hidden, n_output) * cp.sqrt(2. / n_hidden)
            self.b2 = cp.zeros((1, n_output))
        else:
            self.W1 = np.random.randn(
                n_input, n_hidden) * np.sqrt(2. / n_input)
            self.b1 = np.zeros((1, n_hidden))
            self.W2 = np.random.randn(
                n_hidden, n_output) * np.sqrt(2. / n_hidden)
            self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        """Forward pass through the network"""
        z1 = X @ self.W1 + self.b1
        if self.use_gpu:
            a1 = cp.tanh(z1)
        else:
            a1 = np.tanh(z1)

        z2 = a1 @ self.W2 + self.b2

        if self.use_gpu:
            y_pred = 1 / (1 + cp.exp(-z2))  # sigmoid
        else:
            y_pred = 1 / (1 + np.exp(-z2))  # sigmoid

        return y_pred, a1

    def backward(self, X, y_true, y_pred, a1):
        """Compute gradients via backpropagation"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = a1.T @ dz2 / m
        db2 = dz2.sum(axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        if self.use_gpu:
            dz1 = da1 * (1 - cp.power(cp.tanh(a1), 2))
        else:
            dz1 = da1 * (1 - np.power(np.tanh(a1), 2))

        dW1 = X.T @ dz1 / m
        db1 = dz1.sum(axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr=0.1):
        """Update network parameters using gradients"""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, epochs=1000, lr=0.1, batch_size=None, verbose=True):
        """Train the neural network"""
        # Convert data to appropriate device
        X = to_device(X, self.use_gpu)
        y = to_device(y, self.use_gpu)

        # Store loss history
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            y_pred, a1 = self.forward(X)

            # Compute loss
            loss = binary_cross_entropy(y_pred, y, self.use_gpu)
            loss_history.append(to_cpu(loss, self.use_gpu))

            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred, a1)

            # Update parameters
            self.update_params(dW1, db1, dW2, db2, lr)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {to_cpu(loss, self.use_gpu):.4f}")

        return loss_history

    def predict(self, X):
        """Make predictions with the trained model"""
        X = to_device(X, self.use_gpu)
        y_pred, _ = self.forward(X)
        return to_cpu(y_pred, self.use_gpu)

    def generate_decision_boundary(self, X, y, h=0.01):
        """Generate mesh grid for plotting decision boundary"""
        X_np = to_cpu(X, self.use_gpu)
        y_np = to_cpu(y, self.use_gpu)

        # Set min and max values with some margin
        x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
        y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5

        # Generate a grid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Predict class for each point in the mesh
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        return xx, yy, Z


def generate_data(n_samples=1000, noise=0.2, random_state=42):
    """Generate toy data for binary classification"""
    X, y = make_moons(n_samples=n_samples, noise=noise,
                      random_state=random_state)
    X = X.astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return X, y


def plot_data(X, y):
    """Plot the dataset"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
    plt.title("Toy Data - Binary Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    return plt.gcf()


def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of the trained model"""
    plt.figure(figsize=(10, 6))
    xx, yy, Z = model.generate_decision_boundary(X, y)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)

    # Plot the original data points
    X_np = to_cpu(X, model.use_gpu)
    y_np = to_cpu(y, model.use_gpu)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np.ravel(),
                cmap='coolwarm', edgecolors='k')

    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    return plt.gcf()


def plot_training_loss(loss_history_cpu, loss_history_gpu=None):
    """Plot the training loss curves for comparison"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_cpu, label='CPU Training', color='#1E88E5')

    if loss_history_gpu is not None:
        plt.plot(loss_history_gpu, label='GPU Training', color='#FF0D57')

    plt.title('Training Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()
