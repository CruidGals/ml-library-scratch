import numpy as np
import yaml
import os
from modules import Module

def get_state_dict(model: list[Module]) -> dict:
    """ Save the parameters of the model to a file """
    # Initialize the parameters list
    state_dict = {}

    # Get parameters for each module
    for i, module in enumerate(model):
        params = module.get_params()
        state_dict[f"layer_{i}"] = {param.name: param.value for param in params}
    
    return state_dict

def save_state_dict(model: list[Module], file_path: str) -> None:
    """ Save the parameters of the model to a file """
    state_dict = get_state_dict(model)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, state_dict)

def load_state_dict(modules: list[Module], file_path: str) -> None:
    """ Load the parameters of the model from a file """
    # Load the state dictionary
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    state_dict = np.load(file_path, allow_pickle=True).item()

    # Load the parameters for each module
    for i, module in enumerate(modules):
        params = state_dict[f"layer_{i}"].item()
        module.load_params(params) 

def get_modules_info(model: list[Module], file_path: str) -> None:
    """ Get the information of the modules and save as a yaml file """
    module_info = {"modules": []}

    for module in model:
        module_info["modules"].append(module.get_info())

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(module_info, f)

def load_modules_info(file_path: str) -> list[Module]:
    """ Load the information of the modules from a yaml file """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'r') as f:
        module_info = yaml.load(f, Loader=yaml.FullLoader)

    modules = []
    for module in module_info["modules"]:
        if module["name"] == "Linear":
            modules.append(Linear(module["in_neurons"], module["out_neurons"]))
        elif module["name"] == "Conv2D":
            modules.append(Conv2D(module["input_size"], module["in_channels"], module["out_channels"], module["kernel_size"], module["stride"], module["padding"]))
        elif module["name"] == "MaxPool2D":
            modules.append(MaxPool2D(module["kernel_size"], module["stride"], module["padding"]))
        elif module["name"] == "ReLU":
            modules.append(ReLU())
        elif module["name"] == "Sigmoid":
            modules.append(Sigmoid())