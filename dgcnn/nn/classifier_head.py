from dgcnn.nn.util import Sequential
from torch.nn import Linear, LayerNorm, BatchNorm1d, SiLU, Dropout


class ClassifierHead(Sequential):
    def __init__(self, num_classes: int, embedding_features=1024):
        super().__init__(*[
            # todo: is this layer norm beneficial?
            LayerNorm(embedding_features),
            Linear(embedding_features, 512),
            LayerNorm(512),
            SiLU(),
            Dropout(0.5),
            Linear(512, 256),
            LayerNorm(256),
            SiLU(),
            Dropout(0.5),
            Linear(256, num_classes),
        ])
