import torch


class AppendNormals(torch.nn.Module):
    def forward(self, pos, normal=None):
        # Conditionally append normals if present
        if normal is not None:
            return torch.cat([pos, normal], dim=-1)
        else:
            return pos
