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

class Conv2D(Module):
    def __init__(self, input_size: np.ndarray, in_channels: int, out_channels: int, kernel_size: np.ndarray, stride: np.ndarray = np.array([1,1]), padding: np.ndarray = np.zeros(2)):
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Computed using specific equation
        self.output_size = (out_channels,
                            int(np.floor((input_size[0] - kernel_size[0] + 2 * padding[0] + stride[0]) / stride[0])),
                            int(np.floor((input_size[1] - kernel_size[1] + 2 * padding[1] + stride[1]) / stride[1])))

        # Make sure that kernel, stride, and padding is valid
        self.validate_parameters(input_size, kernel_size, stride, padding)

        # Make a weights vector same as kernel size + size is (out_channel, in_channel, kernel_height, kernel_width)
        self.kernels = Parameter(np.zeros((out_channels, in_channels, kernel_size[0], kernel_size[1])), "k")
        self.b = Parameter((self.out_channels, 1, 1), "b")

        # Initiliaze default with normal distribution
        self.initialize_parameters(init.normal)
    
    def validate_parameters(self, input_size, kernel_size, stride, padding):
        # Add the padding to input_size
        padded_input = input_size + 2 * padding

        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("Kernel must be positive")
        
        if stride[0] <= 0 or stride[1] <= 0:
            raise ValueError("Stride must be positive")
        
        if kernel_size[0] > padded_input[0] or kernel_size[1] > padded_input[1]:
            # Raise for now
            raise ValueError("Kernel is larger than input size")

    def initialize_parameters(self, init_method):
        """ Initializes variables with given init method """
        init_method(self.kernels.value)
        init_method(self.b.value)

    def conv2d(self, X):
        """ Perform the convolution """
        # Mental note, X is in shape (batch_size, input_chan, in_h, in_w)
        # Pad input
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))

        # Prepare the output
        batch_size = X.shape[0]
        out_channels, output_height, output_width = self.output_size
        Z = np.zeros((batch_size, out_channels, output_height, output_width))

        for i in range(self.output_size[1]):
            for j in range(self.output_size[2]):
                row_idx = (self.stride[0] * i, self.stride[0] * i + self.kernel_size[0])
                col_idx = (self.stride[1] * j, self.stride[1] * j + self.kernel_size[1])

                # Get sub arr from input
                X_sub = X_padded[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]

                # Compute convolution and save to Z
                Z[:,:,i,j] = np.tensordot(X_sub, self.kernels, axes=([1, 2, 3], [1, 2, 3]))

        return Z

    def forward(self, X):
        # Save input
        self.X = X
        
        # Perform computation
        self.z = self.conv2d(X) + self.b.value

        return self.z
    
    def predict(self, X):
        return self.conv2d(X) + self.b.value

    def backward(self, grad_z):
        # Grad z in the shape of (batch_size, out_channels, out_h, out_w)
        # Pad the input for later
        X_padded = np.pad(self.X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))

        # Compute the intermediates for local gradient
        grad_z_exp = grad_z[:, :, np.newaxis, :, :, np.newaxis, np.newaxis]
        kernels_exp = self.kernels[np.newaxis, :, :, np.newaxis, np.newaxis, :, :]
        scaled_kernels = grad_z_exp * kernels_exp # Shape is (batch, out, in, out_h, out_w, k_h, k_w)

        # Add them in correctly during the loop of kernel gradient
        padded_local_grad = np.zeros(X_padded.shape)

        # Calculate the kernel gradient
        # each local gradient element by the inputs that created the z
        self.kernels.grad = np.zeros(self.kernels.value.shape)
        for i in range(self.output_size[1]):
            for j in range(self.output_size[2]):
                row_idx = (self.stride[0] * i, self.stride[0] * i + self.kernel_size[0])
                col_idx = (self.stride[1] * j, self.stride[1] * j + self.kernel_size[1])

                # Compute the gradient at those specific set of inputs
                X_sub = X_padded[:, np.newaxis, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
                self.kernels.grad += np.einsum('bijk,bo->oijk', X_sub, grad_z[:, :, i, j])

                # self.kernels.grad += np.sum(X_sub * grad_z[:, :, i, j, np.newaxis, np.newaxis, np.newaxis], axis=0)

                # # Copute the local grad
                # current_patch = scaled_kernels[:, :, :, i, j, :, :]
                # patch_grad = np.sum(current_patch, axis=1)
                # padded_local_grad[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]] += patch_grad;

                # Better method of computing gradient (saves a ton of ram)
                patch_grad = np.einsum('oijk,bo->bijk', self.kernels.value, grad_z[:, :, i, j])
                padded_local_grad[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]] += patch_grad

        # Constrain local_grad (remove the padding)
        h_slice = slice(self.padding[0], -self.padding[0]) if self.padding[0] > 0 else slice(None)
        w_slice = slice(self.padding[1], -self.padding[1]) if self.padding[1] > 0 else slice(None)
        self.local_grad = padded_local_grad[:, :, h_slice, w_slice]

        self.b.grad = np.sum(grad_z, axis=(0,2,3))

        return self.local_grad
    
    def get_params(self) -> list[Parameter]:
        """ Return all the parameters for optimization """
        return [self.kernels, self.b]
    
    def zero_grad(self):
        self.local_grad = None

        self.kernels.grad = np.zeros_like(self.kernels.value)
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