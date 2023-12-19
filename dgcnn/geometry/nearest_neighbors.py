import torch
import torch_cluster


def variable_n_nearest_k_neighbors(positions, batch, k=20):
    return torch_cluster.knn_graph(
        positions.float(), k=k, batch=batch,
        loop=True, flow='target_to_source'
    )[1].reshape(-1, k)


class FindNearestNeighbors(torch.nn.Module):
    def __init__(self, k=20, d=None):
        super().__init__()
        self.k = k
        self.d = d

    def forward(self, x, batch):
        positions = x if self.d is None else x[:, :self.d].contiguous()
        neighbor_indices = variable_n_nearest_k_neighbors(positions, batch, k=self.k)
        return {"neighbor_indices": neighbor_indices}


def collect_edge_features(x, neighbor_indices):
    return x[neighbor_indices, :]


class CollectEdgeFeatures(torch.nn.Module):
    def forward(self, x, neighbor_indices):
        return collect_edge_features(x, neighbor_indices)
