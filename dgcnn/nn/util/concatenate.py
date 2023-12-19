import torch
from .sequential import invoke_with_kwargs


class Concatenate(torch.nn.ModuleList):

    def __init__(self, *modules: torch.nn.Module, dim=-1):
        super().__init__(modules)
        self.dim = dim

    def forward(self, *args, **kwargs):
        results = []
        for module in self:
            results.append(invoke_with_kwargs(module, *args, **kwargs))
        return torch.cat(results, dim=self.dim)


class SequentialWithConcatenatedResults(torch.nn.ModuleList):
    def __init__(self, *modules, dim=-1):
        super().__init__(modules)
        self.dim = dim

    def forward(self, *args, **kwargs):
        results = []
        for module in self:

            result = invoke_with_kwargs(module, *args, **kwargs)

            if type(result) is torch.Tensor:
                args = [result]
            elif type(result[-1]) is dict:
                *args, new_kwargs = result
                kwargs = kwargs | new_kwargs
            else:
                args = result

            results.append(args[0])

        out = torch.cat(results, dim=self.dim)
        return out
