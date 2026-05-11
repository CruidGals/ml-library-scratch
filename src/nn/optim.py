import numpy as np
from modules import Module, Parameter

class Adam:
    def __init__(self, modules: list[Module], learning_rate=1e-4, weight_decay=0, beta1=0.9, beta2=0.999):
        self.modules: list[Module] = modules
        self.lr = learning_rate
        self.wd = weight_decay # Likely not be using this for now

        # Optimizer special params
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_num = 0

        # With the adam optimizer, initialize the moments for each param
        # Also store parameters for easy lookup again
        self.model_params: list[Parameter] = []
        for module in self.modules:
            params = module.get_params()

            # No params
            if len(params) == 0:
                continue

            for param in params:
                param.moment1 = np.zeros_like(param.value)
                param.moment2 = np.zeros_like(param.value)
                self.model_params.append(param)

    def zero_grad(self):
        """ Zeroes the gradient for each module """
        for module in self.modules:
            module.zero_grad()

    def step(self):
        """ Steps through each module and updates parameters """

        # Increment the step counter (used to unbias)
        self.step_num += 1

        for param in self.model_params:
            grad = param.grad + self.wd * param.value

            # Perform adam step (do inplace operations to save memory)
            param.moment1 *= self.beta1
            param.moment1 += (1 - self.beta1) * grad

            param.moment2 *= self.beta2
            param.moment2 += (1 - self.beta2) * grad * grad

            # Unbias the moments to precent slow starts
            moment1_unbias = param.moment1 / (1 - self.beta1 ** self.step_num)
            moment2_unbias = param.moment2 / (1 - self.beta2 ** self.step_num)

            # Finally update the parameter
            param.value -= (self.lr * moment1_unbias) / (np.sqrt(moment2_unbias) + 1e-7)
