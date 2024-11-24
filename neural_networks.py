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
        self.activation_fn = np.tanh if activation == 'tanh' else np.relu  # Activation function
        self.activation_fn_grad = lambda x: 1 - np.tanh(x) ** 2 if activation == 'tanh' else np.heaviside(x, 0)

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Store gradients for visualization
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None

    def forward(self, X):
        # Forward pass: Input -> Hidden -> Output
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation_fn(self.z1)  # Hidden layer activations
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = np.tanh(self.z2)  # Output activation (for simplicity)

        return self.out

    def backward(self, X, y):
        # Compute gradients using backpropagation
        error = self.out - y  # Error at output
        d_out = error * (1 - np.tanh(self.z2) ** 2)

        self.grad_W2 = self.a1.T @ d_out
        self.grad_b2 = np.sum(d_out, axis=0, keepdims=True)

        d_hidden = d_out @ self.W2.T * self.activation_fn_grad(self.z1)
        self.grad_W1 = X.T @ d_hidden
        self.grad_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.lr * self.grad_W2
        self.b2 -= self.lr * self.grad_b2
        self.W1 -= self.lr * self.grad_W1
        self.b1 -= self.lr * self.grad_b1

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

    # Plot hidden features
    hidden_features = mlp.a1  # Activations in the hidden layer
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)

    # Hyperplane visualization in the hidden space
    xx, yy = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    z = (-mlp.W2[0, 0] * xx - mlp.W2[1, 0] * yy - mlp.b2[0, 0]) / mlp.W2[2, 0]
    ax_hidden.plot_surface(xx, yy, z, alpha=0.3, color='tan')

    ax_hidden.set_title("Hidden Space")
    ax_hidden.set_xlabel("h1")
    ax_hidden.set_ylabel("h2")
    ax_hidden.set_zlabel("h3")

    # Plot decision boundary in the input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap='bwr')
    ax_input.set_title("Input Space")
    ax_input.set_xlabel("x1")
    ax_input.set_ylabel("x2")

    # Visualize gradients as circles and edges
    for i in range(mlp.W1.shape[1]):  # Loop over hidden neurons
        circle = Circle((0.5, i * 0.3), radius=0.05, color='blue', alpha=0.6)
        ax_gradient.add_patch(circle)
        ax_gradient.arrow(0.5, i * 0.3, mlp.grad_W1[0, i] * 0.1, mlp.grad_W1[1, i] * 0.1,
                          head_width=0.02, color='purple', alpha=0.6)

    ax_gradient.set_title("Gradient Visualization")
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(-0.1, 1)

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
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), 
                        frames=step_num // 10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
