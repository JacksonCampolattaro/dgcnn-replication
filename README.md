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

To reduce boilerplate, the training loop is implemented with the help of [Pytorch Lightning](https://lightning.ai/).
This simplifies logging and ensures that the best available hardware is used.

## Material Differences

The network contains several structural differences from DGCNN as it was described in the original paper.

#### Additional Regularization

The original DGCNN used a fixed number of points (1024, in the case of the primary ModelNet test).
This implementation uses a dynamic point count, with the help of the batch_index pattern common in networks built
with `torch_geometric`.

Not only should this theoretically improve generalization and robustness to noise
(see: [Noise Injection-based Regularization](https://arxiv.org/abs/2103.15027)),
it also has the benefit of significantly increasing training size,
because most batches will contain far fewer than the maximum of 1024*b points.
This is especially important for DGCNN, because the KNN computation can be extremely expensive for high point counts.

#### Moved BatchNorm

Many implementations of DGCNN perform BatchNorms inside the EdgeMLP layers.
This is a more expensive operation when the number of points-per-batch is variable.
We instead apply the BatchNorm _after_ the EdgeMLP, but before the max-pooling.
This should produce a similar effect, because the EdgeMLPs used in DGCNN only have one layer,
and BatchNorms shouldn't change the max features in pooling.

#### Modern Classifier Head

The classifier head has a few changes which make it more similar to contemporary networks,
these changes showed minor improvements in convergence speed during testing:

- Leaky ReLU has been replaced with SiLU
- BatchNorm has been replaced with LayerNorm

## Hyperparameters

|          Parameter | Setting          |
|-------------------:|:-----------------|
|         Batch size | 64               |
|          Optimizer | AdamW            |
| Base Learning Rate | 1E-3             |
|       Lr-scheduler | Cosine Annealing |
|    # of Points (N) | 1024             |
| # of Neighbors (k) | 20               |
|        # of Epochs | 250              |

## Results

#### Performance

[KeOps](https://www.kernel-operations.io/keops/index.html) has large performance benefits for 3d KNN search,
but this advantage is smaller in higher dimensions, such as those found in the later layers of DGCNN.
Because the KNN search accounts for >45% of runtime, this can still produce a useful reduction in runtim per epoch:

| KNN Implementation | Time per Epoch (s) |
|-------------------:|:------------------:|
|      Torch-cluster |       15.16        |
|              KeOps |       12.76        |

#### Accuracy

(todo)
