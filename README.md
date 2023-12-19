# dgcnn-replication

This repository aims to replicate certain results from the
paper [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).
We build a classifier similar to the original DGCNN, and train it on the ModelNet40 dataset.

On nvidia machines, requirements can be installed with:

```bash
pip install -r requirements.txt
```

Running the network is as simple as the following:

```bash
python3 modelnet_classification.py
```

By default, the settings attempt to replicate the results in table 2 of the paper,
but more configuration is available with `python3 modelnet_classification.py -h`

A live view of the network's training progress is available with [aim](https://github.com/aimhubio/aim).
To see this, use:

```bash
aim up
```

and visit the URL of the locally hosted page.

## Network Design

One goal of this reimplementation is the express the structure of DGCNN in a way which closely matches the description
produced by `torchinfo`, as shown below.

```
=====================================================================================
Layer (type:depth-idx)                                       Param #
=====================================================================================
DGCNNClassifier                                              --
├─Sequential: 1-1                                            --
│    └─DynamicEdgeConv: 2-1                                  --
│    │    └─AppendNormals: 3-1                               --
│    │    └─SequentialWithConcatenatedResults: 3-2           --
│    │    │    └─DynamicEdgeConvBlock: 4-1                   --
│    │    │    │    └─FindNearestNeighbors: 5-1              --
│    │    │    │    └─CollectEdgeFeatures: 5-2               --
│    │    │    │    └─Concatenate: 5-3                       --
│    │    │    │    │    └─Identity: 6-1                     --
│    │    │    │    │    └─CentralizeEdgeFeatures: 6-2       --
│    │    │    │    └─EdgeMLP: 5-4                           --
│    │    │    │    │    └─EdgeMLPBlock: 6-3                 384
│    │    │    │    │    └─MaxPool: 6-4                      --
│    │    │    │    │    └─BatchNorm1d: 6-5                  128
│    │    │    └─DynamicEdgeConvBlock: 4-2                   --
│    │    │    │    └─FindNearestNeighbors: 5-5              --
│    │    │    │    └─CollectEdgeFeatures: 5-6               --
│    │    │    │    └─Concatenate: 5-7                       --
│    │    │    │    │    └─Identity: 6-6                     --
│    │    │    │    │    └─CentralizeEdgeFeatures: 6-7       --
│    │    │    │    └─EdgeMLP: 5-8                           --
│    │    │    │    │    └─EdgeMLPBlock: 6-8                 8,192
│    │    │    │    │    └─MaxPool: 6-9                      --
│    │    │    │    │    └─BatchNorm1d: 6-10                 128
│    │    │    └─DynamicEdgeConvBlock: 4-3                   --
│    │    │    │    └─FindNearestNeighbors: 5-9              --
│    │    │    │    └─CollectEdgeFeatures: 5-10              --
│    │    │    │    └─Concatenate: 5-11                      --
│    │    │    │    │    └─Identity: 6-11                    --
│    │    │    │    │    └─CentralizeEdgeFeatures: 6-12      --
│    │    │    │    └─EdgeMLP: 5-12                          --
│    │    │    │    │    └─EdgeMLPBlock: 6-13                16,384
│    │    │    │    │    └─MaxPool: 6-14                     --
│    │    │    │    │    └─BatchNorm1d: 6-15                 256
│    │    │    └─DynamicEdgeConvBlock: 4-4                   --
│    │    │    │    └─FindNearestNeighbors: 5-13             --
│    │    │    │    └─CollectEdgeFeatures: 5-14              --
│    │    │    │    └─Concatenate: 5-15                      --
│    │    │    │    │    └─Identity: 6-16                    --
│    │    │    │    │    └─CentralizeEdgeFeatures: 6-17      --
│    │    │    │    └─EdgeMLP: 5-16                          --
│    │    │    │    │    └─EdgeMLPBlock: 6-18                65,536
│    │    │    │    │    └─MaxPool: 6-19                     --
│    │    │    │    │    └─BatchNorm1d: 6-20                 512
│    │    └─PointMLP: 3-3                                    --
│    │    │    └─PointMLPBlock: 4-5                          --
│    │    │    │    └─Linear: 5-17                           524,288
│    │    │    │    └─LeakyReLU: 5-18                        --
│    │    └─VNMaxMeanPool: 3-4                               --
│    │    │    └─VNMaxPool: 4-6                              --
│    │    │    └─VNMeanPool: 4-7                             --
│    └─ClassifierHead: 2-2                                   --
│    │    └─LayerNorm: 3-5                                   4,096
│    │    └─Linear: 3-6                                      1,049,088
│    │    └─LayerNorm: 3-7                                   1,024
│    │    └─SiLU: 3-8                                        --
│    │    └─Dropout: 3-9                                     --
│    │    └─Linear: 3-10                                     131,328
│    │    └─LayerNorm: 3-11                                  512
│    │    └─SiLU: 3-12                                       --
│    │    └─Dropout: 3-13                                    --
│    │    └─Linear: 3-14                                     10,280
├─CrossEntropyLoss: 1-2                                      --
├─MulticlassAccuracy: 1-3                                    --
├─MulticlassAccuracy: 1-4                                    --
=====================================================================================
Total params: 1,812,136
Trainable params: 1,812,136
Non-trainable params: 0
=====================================================================================
```

Pytorch is surprisingly suitable for this sort of declarative layout, with the help of a custom `Sequential` module.

For example, the classifier head implementation is very straightforward:

```python
class ClassifierHead(Sequential):
    def __init__(self, num_classes: int, embedding_features=1024):
        super().__init__(*[
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
```

## Material Differences

The network contains several structural differences from DGCNN as described in the original paper.

#### Additional Regularization

(todo)

#### No BatchNorm

(todo)

## Results

(todo)
