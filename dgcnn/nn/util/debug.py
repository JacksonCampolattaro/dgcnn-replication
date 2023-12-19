import torch

class DebugPrintShape(torch.nn.Module):
    def forward(self, *args: torch.Tensor, **kwargs):
        print(
            *[x.shape for x in args],
            *[f"{name}:{x.shape}" for name, x in kwargs.items()],
        )
        return *args, kwargs
