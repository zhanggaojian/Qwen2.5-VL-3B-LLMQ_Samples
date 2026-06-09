import importlib
import os
import sys

import torch


def load_pytorch_model(model_name: str, path: str, filename: str, load_state_dict: bool = False) -> torch.nn.Module:
    """
    Load the pytorch model from the given path and filename.
    NOTE: The model can only be saved by saving the state dict. Attempting to serialize the entire model will result
    in a mismatch between class types of the model defined and the class type that is imported programatically.

    :param model_name: Name of model
    :param path: Path where the pytorch model definition file is saved
    :param filename: Filename of the pytorch model definition
    :param load_state_dict: If True, load state dict with the given path and filename. The state dict file is expected
        to end in '.pth'
    :return: Imported pytorch model
    """

    model_path = os.path.join(path, filename + '.py')
    if not os.path.exists(model_path):
        # logger.error('Unable to find model file at path %s', model_path)
        raise AssertionError('Unable to find model file at path ' + model_path)

    # Import model's module and instantiate model
    spec = importlib.util.spec_from_file_location(filename, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[filename] = module
    spec.loader.exec_module(module)
    model = getattr(module, model_name)()

    # Load state dict if necessary
    if load_state_dict:
        state_dict_path = os.path.join(path, filename + '.pth')
        if not os.path.exists(state_dict_path):
            # logger.error('Unable to find state dict file at path %s', state_dict_path)
            raise AssertionError('Unable to find state dict file at path ' + state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))

    return model