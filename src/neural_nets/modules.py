import numpy as np
import param_init as init

# Implementation inspired by PyTorch

# Have a class that isolates parameter weights and updates
class Parameter:
    def __init__(self, value: np.ndarray, name=""):
        self.name = name
        self.value = value

        # For gradient
        self.grad: np.ndarray | None = None

        # Any other variables required for optimizatin will be defined by the optimizer

class Module:
    def __init__(self):
        # Keep these intermediates for each module
        self.X: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.local_grad: np.ndarray | None = None

    def forward(self, X):
        """ Forward propogates from this function """
        raise NotImplementedError
    
    def predict(self, X):
        """ Forward propogation but don't save intermediates """

    def backward(self, grad_z):
        """ Backward propogates from this function """
        raise NotImplementedError
    
    def get_params(self) -> list[Parameter]:
        """ Return all the parameters for optimization """
        return []
    
    def zero_grad(self):
        self.local_grad = None

# Main Layers
class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons

        # Init params (in order of gradients)
        self.w: Parameter = Parameter(np.zeros((in_neurons, out_neurons)), "w")
        self.b: Parameter = Parameter(np.zeros(out_neurons), "b")

        # Initiliaze default with normal distribution
        self.initialize_parameters(init.normal)

    def initialize_parameters(self, init_method):
        """ Initializes variables with given init method """
        init_method(self.w.value)
        init_method(self.b.value)

    def forward(self, X):
        self.z = X @ self.w.value + self.b.value

        # Store the intermediate value for later
        self.X = X

        # Return the propogated value
        return self.z
    
    def predict(self, X):
        return X @ self.w.value + self.b.value
    
    def backward(self, grad_z: np.ndarray):
        # Get dz/dx
        self.local_grad = self.w.value.T

        # Get local gradients (use to update later)
        self.w.grad = self.X.T @ grad_z
        self.b.grad = np.sum(grad_z, axis=0)

        # Multiply local grad by upstream grad to get downstream grad
        return grad_z @ self.local_grad
    
    def get_params(self) -> list[Parameter]:
        return [self.w, self.b]

    def zero_grad(self):
        super().zero_grad()

        self.w.grad = np.zeros_like(self.w.value)
        self.b.grad = np.zeros_like(self.b.value)

# Activation Functions
class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        self.z = np.maximum(0,X) # Compute the ReLU function
        return self.z

    def predict(self, X):
        return np.maximum(0,X)
    
    def backward(self, grad_z):
        # Create a mask for gradients
        self.local_grad = (self.X > 0).astype(np.int8)
        
        # Multiply local grad * upstream grad to get downstream grad (elementwise mult)
        return grad_z * self.local_grad

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Save the input for later
        self.X = X

        # Calculate the sigmoid
        self.z = 1 / (1 + np.exp(-X))

        # Return the value
        return self.z
    
    def predict(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, grad_z):
        # Calculate the local gradient for softmax
        self.local_grad = self.z * (1 - self.z)

        # Multiple local grad + upstream gradient to get downstream (element wise)
        return grad_z * self.local_grad