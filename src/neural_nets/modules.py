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

    def load_params(self, params):
        """ Load parameters into the module """

    # For information purposes
    def get_info(self) -> dict:
        """ Get the information of the module """
        raise NotImplementedError
    
    def zero_grad(self):
        self.local_grad = None

# Main Layers
class Linear(Module):
    def __init__(self, in_neurons: int, out_neurons: int):
        super().__init__()
        
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

    def load_params(self, params: dict):
        # Make sure that parameters are in right form
        assert params["w"].shape == (self.in_neurons, self.out_neurons), "Weights must be in the shape of (in_neurons, out_neurons)"
        assert params["b"].shape == self.out_neurons, "Biases must be in the shape of (out_neurons,)"

        # Load in the parameters
        self.w.value = params["w"]
        self.b.value = params["b"]

    def get_info(self) -> dict:
        return {
            "name": "Linear",
            "in_neurons": self.in_neurons,
            "out_neurons": self.out_neurons
        }

    def zero_grad(self):
        super().zero_grad()

        self.w.grad = np.zeros_like(self.w.value)
        self.b.grad = np.zeros_like(self.b.value)

class Conv2D(Module):
    def __init__(self, input_size: np.ndarray, in_channels: int, out_channels: int, kernel_size: np.ndarray, stride: np.ndarray = np.array([1,1]), padding: np.ndarray = np.zeros(2)):
        super().__init__()

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
        self.b = Parameter(np.zeros((self.out_channels, 1, 1)), "b")

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

    def conv2d(self, X_padded, train=True):
        """ Perform the convolution """
        # We will perform convolution using im2col + GEMM
        # Create a sliding window view (numpy), and adjust it for the stride
        patches = np.lib.stride_tricks.sliding_window_view(X_padded, window_shape=(self.kernel_size[0], self.kernel_size[1]), axis=(2, 3))
        patches = patches[:, :, ::self.stride[0], ::self.stride[1]]

        # Reshape to prepare for GEMM
        # want it in the shape (batch_size, out_h, out_w, in_channels * kernel_height * kernel_width)
        batch_size, c, out_h, out_w, k_h, k_w = patches.shape

        if train:
            self.X_col = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, c * k_h * k_w)

            # Perform the convolution with flattened kernel
            Z = self.X_col @ self.kernels.value.reshape(self.out_channels, -1).T
        else:
            patches_flattened = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, c * k_h * k_w)
            Z = patches_flattened @ self.kernels.value.reshape(self.out_channels, -1).T

        # Reshape and return
        return Z.reshape(batch_size, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def forward(self, X):
        # Save and pad input
        self.X_padded = np.pad(X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        
        # Perform computation
        self.z = self.conv2d(self.X_padded) + self.b.value

        return self.z
    
    def predict(self, X):
        # Send the padded input in
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        return self.conv2d(X_padded, train=False) + self.b.value

    def backward(self, grad_z):
        # Grad z in the shape of (batch_size, out_channels, out_h, out_w)
        # Flatten upstream grad to (out_channels, batch_size * out_h * out_w)
        grad_z_flat = grad_z.transpose(1,0,2,3).reshape(self.out_channels, -1)

        # Calculate the kernel gradient
        # (out_channels, batch_size * out_h * out_w) @ (batch_size * out_h * out_w, in_channels * kernel_height * kernel_width) = (out_channels, in_channels * kernel_height * kernel_width)
        self.kernels.grad = (grad_z_flat @ self.X_col).reshape(self.kernels.value.shape)

        # Calculate downstream gradient
        # (batch_size * out_h * out_w, in_channels * kernel_height * kernel_width) @ (in_channels * kernel_height * kernel_width, out_channels) = (batch_size * out_h * out_w, out_channels)
        self.local_grad = grad_z_flat.T @ self.kernels.value.reshape(self.out_channels, -1)
        
        self.b.grad = np.sum(grad_z, axis=(0,2,3)).reshape(self.out_channels, 1, 1)

        return self.col2im(self.local_grad, grad_z.shape)

    def col2im(self, grad_col, grad_z_shape):
        """ Folds 2D column gradients back into a 4D image tensor. """
        batch_size, _, out_h, out_w = grad_z_shape
        # Skip calculating H_pad/W_pad; just use the forward-pass cache
        dX_padded = np.zeros_like(self.X_padded)
        
        # Reshape columns back to (N, out_h, out_w, C_in, k_h, k_w)
        gp = grad_col.reshape(batch_size, out_h, out_w, -1, *self.kernel_size)

        # Flatten the nested kernel loops into one iterator
        for i, j in np.ndindex(*self.kernel_size):
            # Slicing logic: Start at (i, j), step by stride, take exactly out_h/w steps
            # This aligns the (N, C, out_h, out_w) patches back to the padded image
            dX_padded[:, :, i : i + out_h * self.stride[0] : self.stride[0], j : j + out_w * self.stride[1] : self.stride[1]] += gp[..., i, j].transpose(0, 3, 1, 2)

        # Remove the rest of the padding
        p = self.padding[0]
        return dX_padded[:, :, p:-p, p:-p] if p > 0 else dX_padded
    
    def get_params(self) -> list[Parameter]:
        """ Return all the parameters for optimization """
        return [self.kernels, self.b]

    def load_params(self, params: dict):
        # Make sure that parameters are in right form
        assert params["k"].shape == (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]), "Kernels must be in the shape of (out_channels, in_channels, kernel_size[0], kernel_size[1])"
        assert params["b"].shape == (self.out_channels, 1, 1), "Biases must be in the shape of (out_channels, 1, 1)"

        # Load in the parameters
        self.kernels.value = params["k"]
        self.b.value = params["b"]

    def get_info(self) -> dict:
        return {
            "name": "Conv2D",
            "input_size": self.input_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding
        }

    def zero_grad(self):
        self.local_grad = None

        self.kernels.grad = np.zeros_like(self.kernels.value)
        self.b.grad = np.zeros_like(self.b.value)

class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=np.array([1,1]), padding=np.array([0,0])):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # To be cached later
        self.output_size = None

    def forward(self, X: np.ndarray):
        # Save the output
        self.X = X

        # Create the padded input (save the shape for backprop)
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        self.padded_shape = X_padded.shape
            
        # Perform the max pool over all the layers that the kernel passes over
        # Use im2col technique again to make (batch_size, in_channels, out_h, out_w, kernel_height * kernel_width)
        patches = np.lib.stride_tricks.sliding_window_view(X_padded, window_shape=(self.kernel_size[0], self.kernel_size[1]), axis=(2, 3))
        patches = patches[:, :, ::self.stride[0], ::self.stride[1]]

        # Flatten the last two axis (kernel ones) and find the max of those
        batch_size, c, out_h, out_w, k_h, k_w = patches.shape
        patches_flattened = patches.reshape(batch_size, c, out_h, out_w, -1)

        # Get the max pool / max indices vectorized
        self.z = np.max(patches_flattened, axis=-1)
        self.max_indices = np.argmax(patches_flattened, axis=-1)

        return self.z
    
    def predict(self, X: np.ndarray):
        # Create the padded input
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))

        # Same thing as in forward function
        patches = np.lib.stride_tricks.sliding_window_view(X_padded, window_shape=(self.kernel_size[0], self.kernel_size[1]), axis=(2, 3))
        patches = patches[:, :, ::self.stride[0], ::self.stride[1]]

        # Flatten the last two axis (kernel ones) and find the max of those
        batch_size, c, out_h, out_w, k_h, k_w = patches.shape
        patches_flattened = patches.reshape(batch_size, c, out_h, out_w, -1)

        return np.max(patches_flattened, axis=-1)

    def backward(self, grad_z: np.ndarray):
        """ Backward propogation for max pool """

        batch_size, in_channels, out_h, out_w = grad_z.shape
        kernel_height, kernel_width = self.kernel_size
        
        # 1. Create a zeroed matrix for the flattened patches
        # Shape: (batch_size, in_channels, out_h, out_w, kernel_height * kernel_width)
        grad_patches = np.zeros((batch_size, in_channels, out_h, out_w, kernel_height * kernel_width), dtype=grad_z.dtype)

        # 2. Use np.put_along_axis to scatter the gradients
        # We place the grad_z values at the indices stored in self.max_indices
        # max_indices must be shape (N, C, out_h, out_w, 1) for this to work
        np.put_along_axis(grad_patches, self.max_indices[..., np.newaxis].astype(int), grad_z[..., np.newaxis], axis=-1)

        # 3. Reshape and fold back using your optimized col2im
        # Your col2im expects (batch, out_h, out_w, C * k_h * k_w)
        grad_col = grad_patches.transpose(0, 2, 3, 1, 4).reshape(batch_size, out_h, out_w, -1)
        
        return self.col2im(grad_col, grad_z.shape)

    def col2im(self, grad_col, grad_z_shape):
        """ Folds 2D column gradients back into a 4D image tensor. """
        batch_size, _, out_h, out_w = grad_z_shape
        # Skip calculating H_pad/W_pad; just use the forward-pass cache
        dX_padded = np.zeros(self.padded_shape)
        
        # Reshape columns back to (N, out_h, out_w, C_in, k_h, k_w)
        gp = grad_col.reshape(batch_size, out_h, out_w, -1, *self.kernel_size)

        # Flatten the nested kernel loops into one iterator
        for i, j in np.ndindex(*self.kernel_size):
            # Slicing logic: Start at (i, j), step by stride, take exactly out_h/w steps
            # This aligns the (N, C, out_h, out_w) patches back to the padded image
            dX_padded[:, :, i : i + out_h * self.stride[0] : self.stride[0], j : j + out_w * self.stride[1] : self.stride[1]] += gp[..., i, j].transpose(0, 3, 1, 2)

        # Remove the rest of the padding
        p = self.padding[0]
        return dX_padded[:, :, p:-p, p:-p] if p > 0 else dX_padded
    
    def get_info(self) -> dict:
        return {
            "name": "MaxPool2D",
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding
        }

class Flatten(Module):
    def forward(self, X: np.ndarray):
        # Save the input
        self.input_shape = X.shape
        return X.reshape(self.input_shape[0], -1)
    
    def predict(self, X):
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_z: np.ndarray):
        # Just simply restore the shape
        return grad_z.reshape(self.input_shape)
        

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

    def get_info(self) -> dict:
        return {
            "name": "ReLU"
        }

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

    def get_info(self) -> dict:
        return {
            "name": "Sigmoid"
        }

class Dropout(Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self.keep_prob = 1 - self.p

        # Save a generator
        self.rng = np.random.default_rng()

    def forward(self, X: np.ndarray):
        # Create a mask for the dropout
        self.mask = self.rng.binomial(1, self.keep_prob, X.shape)

        # Scale output (inverted dropout)
        return (X * self.mask) / self.keep_prob
    
    def predict(self, X: np.ndarray):
        return X
    
    def backward(self, grad_z: np.ndarray):
        return (grad_z * self.mask) / self.keep_prob
    
    def get_info(self) -> dict:
        return {
            "name": "Dropout", 
            "p": self.p
        }