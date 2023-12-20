import torch
import torch_cluster
from pykeops.torch import LazyTensor


@torch.cuda.amp.autocast(False)
def variable_n_nearest_k_neighbors_keops(positions, batch, k=20, **kwargs):
    # See: https://github.com/getkeops/keops/issues/73
    def diagonal_ranges(batch_x=None, batch_y=None):
        """Encodes the block-diagonal structure associated to a batch vector."""

        def ranges_slices(batch):
            """Helper function for the diagonal ranges function."""
            Ns = batch.bincount()
            indices = Ns.cumsum(0)
            ranges = torch.cat((0 * indices[:1], indices))
            ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
            slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
            return ranges, slices

        if batch_x is None and batch_y is None: return None
        ranges_x, slices_x = ranges_slices(batch_x)
        ranges_y, slices_y = ranges_slices(batch_y)
        return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x

    # fixme: this is slower than torch_geometric's implementation!
    p_i = LazyTensor(positions.float().unsqueeze(-3))
    p_j = LazyTensor(positions.float().unsqueeze(-2))
    d_ij = ((p_i - p_j) ** 2).sum(-1)
    d_ij.ranges = diagonal_ranges(batch, batch)
    indices = d_ij.argKmin(k, dim=1, **kwargs)
    return indices


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
        neighbor_indices = variable_n_nearest_k_neighbors_keops(positions, batch, k=self.k)
        return {"neighbor_indices": neighbor_indices}


def collect_edge_features(x, neighbor_indices):
    return x[neighbor_indices, :]


class CollectEdgeFeatures(torch.nn.Module):
    def forward(self, x, neighbor_indices):
        return collect_edge_features(x, neighbor_indices)
