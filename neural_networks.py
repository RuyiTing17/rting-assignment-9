import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases for input -> hidden and hidden -> output layers
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Store hidden activations
        self.A1 = None

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def forward(self, X):
        self.X = X  # Input
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear transformation for hidden layer
        self.A1 = self.activation(self.z1)  # Apply activation function to hidden layer
        self.z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation for output layer
        self.a2 = self.z2  # For regression (use activation if classification is needed)
        return self.a2
    def backward(self, X, y):
        # Compute the gradient of the loss function (assuming Mean Squared Error)
        loss_grad = 2 * (self.a2 - y) / y.size

        # Gradients for output layer
        dW2 = np.dot(self.A1.T, loss_grad)  # Use self.A1 for the hidden layer activations
        db2 = np.sum(loss_grad, axis=0, keepdims=True)

        # Gradients for hidden layer
        da1 = np.dot(loss_grad, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Hidden features (Assume hidden layer output is stored in `mlp.A1`)
    hidden_features = mlp.A1

    # Plot hidden features in 3D
    if hidden_features.shape[1] == 3:  # Check if hidden features are 3D
        ax_hidden.scatter(
            hidden_features[:, 0],
            hidden_features[:, 1],
            hidden_features[:, 2],
            c=y.ravel(),
            cmap='bwr',
            alpha=0.7
        )

        # Plot hyperplane (decision boundary) in hidden space
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        # Ensure 3D weight structure is valid
        if mlp.W1.shape[0] > 2 and mlp.W1.shape[1] > 0:
            zz = -(mlp.W1[0, 0] * xx + mlp.W1[1, 0] * yy) / mlp.W1[2, 0]
            ax_hidden.plot_surface(xx, yy, zz, alpha=0.2, color='tan')

    elif hidden_features.shape[1] == 2:  # If 2D, skip the 3D plot
        ax_hidden.scatter(
            hidden_features[:, 0],
            hidden_features[:, 1],
            c=y.ravel(),
            cmap='bwr',
            alpha=0.7
        )   

    # Input space decision boundary
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)  # Forward pass for predictions
    ax_input.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')

    # Gradient visualization in network diagram
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    layer_nodes = [2, 3, 1]  # Number of nodes per layer
    positions = []
    for i, num_nodes in enumerate(layer_nodes):
        x = np.linspace(0.2, 0.8, num_nodes)
        y = np.full(num_nodes, 1 - i * 0.5)
        positions.append(list(zip(x, y)))

    # Draw nodes
    for layer in positions:
        for node in layer:
            ax_gradient.scatter(*node, s=200, color='blue')

    # Draw connections with gradients as edge thickness
    for i, (layer1, layer2) in enumerate(zip(positions[:-1], positions[1:])):
        for j, start in enumerate(layer1):
            for k, end in enumerate(layer2):
                grad_magnitude = abs(mlp.W1[j, k]) if i == 0 else abs(mlp.W2[j, k])
                ax_gradient.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color='purple',
                    linewidth=1 + grad_magnitude * 5,
                    alpha=0.7
                )

    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis("off")
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)