import numpy as np
from modules import Module

class Loss:
    def __init__(self, modules: list[Module]):
        # Intermediates
        self.loss: np.ndarray | None = None
        self.preds: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.local_grad: np.ndarray | None = None

        self.modules: list[Module] = modules

    def compute(self, preds, labels):
        # Save intermediates
        self.preds = preds
        self.labels = labels

    def backward(self):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def __init__(self, modules: list[Module]):
        super().__init__(modules)

        # Additional intermediates (store the initial softmax)
        self.softmax: np.ndarray | None = None

    def compute(self, preds: np.ndarray, labels: np.ndarray):
        super().compute(preds, labels)

        # Stabilize samples
        shift_preds = preds - np.max(preds, axis=1, keepdims=True)

        # Calculate the softmax
        exp_preds = np.exp(shift_preds)
        self.softmax = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Compute the loss (use epsilon to avoid log(0) = -inf )
        epsilon = 1e-12
        self.loss = -np.sum(labels * np.log(self.softmax + epsilon), axis=1)

        return self.loss
    
    def backward(self):
        # Using saved softmax, compute local gradient
        self.local_grad = (self.softmax - self.labels) / self.labels.shape[0]

        # Perform backward on the rest of the modules
        grad = self.local_grad
        for module in self.modules[::-1]:
            grad = module.backward(grad)