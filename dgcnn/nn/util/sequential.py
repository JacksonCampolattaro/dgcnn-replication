import inspect
import torch
from torch.nn import Module, ModuleList


def invoke_with_kwargs(module: Module, *args, **kwargs):
    accepts_kwargs = (list(inspect.signature(module.forward).parameters.values())[-1].kind
                      == inspect.Parameter.VAR_KEYWORD)

    if accepts_kwargs:
        return module(*args, **kwargs) if isinstance(args, (tuple, list)) \
            else module(args, **kwargs)
    else:
        parameter_names = list(inspect.signature(module.forward).parameters.keys())
        named_parameters = dict((name, kwargs[name]) for name in parameter_names if name in kwargs)
        return module(*args, **named_parameters) if isinstance(args, (tuple, list)) \
            else module(args, **named_parameters)


class Sequential(ModuleList):
    def __init__(self, *modules: Module):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        for module in self:

            result = invoke_with_kwargs(module, *args, **kwargs)

            if type(result) is torch.Tensor:
                args = [result]
            elif type(result) is dict:
                kwargs = kwargs | result
            elif type(result[-1]) is dict:
                *args, new_kwargs = result
                kwargs = kwargs | new_kwargs
            else:
                args = result

        return args[0] if len(args) == 1 else args
