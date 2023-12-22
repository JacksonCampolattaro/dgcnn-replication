---
title: Replicating DGCNN
author:
  - Hesam Araghi
  - Jackson Campolattaro
layout: post
---

# Replicating DGCNN
*Hesam Araghi and Jackson Campolattaro*

For more information about this project or to try it out yourself, visit
the [GitHub Repo](https://github.com/JacksonCampolattaro/dgcnn-replication).

(todo: contents go here!)

# Reproducibility summary:

# Introduction:

Point clouds represent three-dimensional data in the spatial domain. Examples of point cloud data are LiDAR scanners, which generate spatial information by measuring distances with laser beams, and event camera data output, capturing changes in light intensity at pixel locations over time and converting them into 3D point clouds. Point clouds have various applications in 3D perception tasks, such as object recognition and scene understanding, and point cloud processing can have many applications in tasks such as autonomous navigation, self-driving vehicles, augmented reality, and robotics. 

However, the irregular and unstructured nature of point clouds poses a significant challenge to the application of traditional visual processing methods like convolutional neural networks (CNNs). One common approach to tackle the inherent difficulty in directly applying CNNs to point clouds is converting them into regular grids, such as 3D volumetric representations, to leverage conventional 2D or 3D CNNs. However, this technique can introduce quantization artifacts, leading to an accuracy drop. Additionally, the computational complexity and memory usage associated with transforming point clouds into dense 3D grid structures is high.

In response to these challenges, state-of-the-art methods are designed to operate directly on irregular input data, such as point clouds. These methods need to be invariant to the permutation of the points and can directly extract features from the set of point clouds for subsequent tasks like classification and segmentation. 

One of the popular and widely utilized methods is dynamic graph CNN (DGCNN). Its approach involves computing embeddings for edges connected to a node and aggregating them in a permutation-invariant way to extract node features. Moreover, the connecting graph between the points is constructed dynamically in each layer of the network. 

In this reproducibility project, we opted to re-implement the DGCNN method and assess its performance in the shape classification task on the ModelNet40 dataset. We followed the “replicated” criteria, choosing to implement the DGCNN from scratch rather than relying on existing libraries like PyTorch Geometric. The reason was to gain a good understanding of the method's intricacies and its implementation details. Additionally, we used matrix and tensor operations to implement the algorithm instead of message passing used in the PyTorch Geometric library, aiming to potentially enhance execution speed, although it may come at the cost of reducing some flexibility in the algorithm.


# Method:

In this section, we want to explain the working mechanism of the DGCNN method briefly. The distinguishing feature of the DGCNN method, setting it apart from previous approaches like PointNet, is that, instead of operating on individual points and aggregating their feature vectors, DGCNN generates an underlying graph on the points and performs convolution operations on features derived from the edges connecting neighboring points. Each layer is thus named "EdgeConv" to signify the convolution of edge features. Another difference from earlier methods is the dynamic nature of the graph in DGCNN. Unlike fixed graphs in previous models, the graph changes at each network layer. This dynamic graph is achieved by computing the k-nearest neighbors on the embeddings of the points (nodes of the graph) at each layer. In the following subsections, we explain edge convolution and dynamic graph updates.

## Edge convolution

EdgeConv takes as input a set of points and their corresponding features and computes the output features for each point. Each input point is represented by features associated with that point (such as its spatial coordinates, color, or intensity). Let $$\mathcal{G}=(\mathcal{V},\mathcal{E})$$ be a directed graph, where $$\mathcal{V}$$ is the set of graph nodes with the size of $$\lvert \mathcal{V}\rvert = N$$, and $$\mathcal{E} \subseteq \mathcal{V}\times \mathcal{V}$$ is the set of graph edges. Suppose, we have $$N$$ points which are represented by the nodes of the graph $$\mathcal{G}$$, and node input and output features are denoted by sets of vectors $$\lbrace\mathbf{x}^{(in)}_i\rbrace_{i=1}^{N}\subseteq \mathbb{R}^{F_{in}}$$ and $$\lbrace\mathbf{x}^{(out)_i}\rbrace_{i=1}^{N}\subseteq \mathbb{R}^{F_{out}}$$, respectively. For each edge $$(i,j)\in\mathcal{E}$$, the edge features are defined as $$\mathbf{e}_{ij}=h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})$$, where function $$h_{\mathbf{\Theta}}:\mathbb{R}^{F_{in}}\times\mathbb{R}^{F_{in}}\rightarrow \mathbb{R}^{F_{out}}$$ maps the features of two neighboring nodes to the connecting edge features. The output features of node $$i$$ in EdgeConv layer can be obtained as

$$
\mathbf{x}^{(out)}_i =\mathop{\square}_{j\in \mathcal{N}_i} h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)}),
$$

where the operator $$\square$$ defines an aggregation operation that is permutation invariant. Examples of such operators are *summation* and *maximization*. The set $$\mathcal{N}_i\subseteq\mathcal{V}$$ define the neighboring nodes for the node $$i$$. 

For this paper, the authors choose the function $$h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})$$ as follows:

$$
h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})={\bar h}_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)} - \mathbf{x}_i^{(in)}),
$$

where $${\bar h}_{\mathbf{\Theta}}$$ is a multilayer perceptron network (MLP) with an input channel size of $$2\,F_{in}$$  and output channel size of $$F_{out}$$, and the parameters of the network is indicated by $$\mathbf{\Theta}$$. The input to the MLP is the concatenated vector of $$\mathop{concat}\big({\mathbf{x}_i^{(in)}},\mathbf{x}_j^{(in)} - \mathbf{x}_i^{(in)}\big)\in\mathbb{R}^{2\,F_{in}}$$. The authors suggest that using relative feature vectors of 
$$\mathbf{x}_j^{(in)} - \mathbf{x}_i^{(in)}$$ can better capture the local neighborhood information. 

## Dynamic graph update

For computing the EdgeConv layer, we need to define the graph $$\mathcal{G}=(\mathcal{V},\mathcal{E})$$. The paper suggests updating the graph *dynamically* based on the feature vectors at each layer. 
Particularly, there is a graph $$\mathcal{G}^{(l)}=(\mathcal{V}^{(l)},\mathcal{E}^{(l)})$$ at layer $l$'th of the network that is constructed by performing $$k$$-nearest neighboring (kNN) algorithm on input features $$\mathbf{x}_1^{(in)},\ldots,\mathbf{x}_N^{(in)}$$.

This dynamic graph update distinguishes this work from the graph CNNs methods that rely on fixed graphs constructed from the input data. 
The authors mentioned that dynamic graph update enables the network to aggregate features from spatially distant points without making the network very deep or constructing a very dense graph (e.g. fully connected graph). However, the dynamic graph update can impose very high computational complexity during both training and model inference. 
In contrast to fixed graphs, where the graph is computed once per data sample and used throughout training epochs, in the dynamic scenario, during each forward pass of the network, the graph needs to be computed a number of times equal to the number of layers.
This would be more severe especially when dealing with a larger number of points, as the complexity of computing the k-nearest neighbors (kNN) algorithm grows quadratically with the number of points.

# Experiment results

In this section, we present the results of our re-implemented of the DGCNN algorithm in the context of the *classification* task on the ModelNet40 dataset. We assess the classification accuracy with the results reported in the original paper. Additionally, we evaluate the efficiency of two distinct implementations of the k-nearest neighbors (kNN) method by measuring the training's running time.

## Dataset 

We used the ModelNet40 dataset \[2\] for point cloud classification. It contains 12,311 CAD models across 40 object categories like chairs, tables, and lamps. 
Similar to the paper, we have the following settings for the dataset:
- 9,843 models for training and 2,468 models for testing
- We [normalize](https://github.com/JacksonCampolattaro/dgcnn-replication/blob/7e9c3935375b89ed1262b533c58d9e17a9bd447c/dgcnn/data/modelnet_datamodule.py#L25) the point cloud to fit in the unit sphere.
- We [sample](https://github.com/JacksonCampolattaro/dgcnn-replication/blob/7e9c3935375b89ed1262b533c58d9e17a9bd447c/dgcnn/data/modelnet_datamodule.py#L26) uniformly 1024 points from the surfaces of the mesh faces. 
- The following augmentations are performed during the training:
	- Randomly [scaling](https://github.com/JacksonCampolattaro/dgcnn-replication/blob/7e9c3935375b89ed1262b533c58d9e17a9bd447c/dgcnn/data/modelnet_datamodule.py#L31) the objects within the scales (2/3,3/2).
	- Randomly [shift](https://github.com/JacksonCampolattaro/dgcnn-replication/blob/7e9c3935375b89ed1262b533c58d9e17a9bd447c/dgcnn/data/modelnet_datamodule.py#L33) the objects by the maximum offset of 0.2.
	- Randomly [drop](https://github.com/JacksonCampolattaro/dgcnn-replication/blob/7e9c3935375b89ed1262b533c58d9e17a9bd447c/dgcnn/data/modelnet_datamodule.py#L35) the points in the objects with the maximum ratio of 0.875.

The original DGCNN used a fixed number of points while this implementation uses a random drop augmentation which can improve generalization and robustness to noise (see \[3\]). This can also help reduce the computation cost during the training because the KNN would usually be performed on a smaller number of points.

## Network architecture

The network consists of two parts: the feature extractor and the classification head.
For the input feature, we concatenate the 3-dimensional position vector and surface normals, thus 
$$F_{in} = 6$$.
Then, the input points are given to the feature extractor part which has 4 EdgeConv layers. In each layer, first the graph is constructed using k-NN algorithm with 
$$k=20$$.
For implementing the k-NN algorithm, we tested two different implementations: 
1) [torch-cluster](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.knn.html?highlight=knn#torch-geometric-nn-pool-knn) which is a fast implementation of k-NN on GPUs,
2) [KeOps](https://www.kernel-operations.io/keops/index.html) \[4\] which is a library to speed up the computation of the reductions in large arrays in geometric applications.
Then the edge features are created using an MLP network with output feature dimensions of (64, 64, 128, 256) for layers 1 to 4. Then, the edge features are aggregated with the 'max' operator.
Then, the output features of all the layers are concatenated into a 64+64+128+256=512 feature vector. Then, an MLP network is applied to 512-dimensional features of each point to generate 1024-length feature vectors. We compute then a  global maximum on the whole feature vectors of the points.

For the classification head, we used a 3-layer MLP network with hidden layer sizes of (512, 256, 40) with layer normalization, 50 percent dropout, and Sigmoid Linear Units (SiLUs) for the activation functions.

The detailed differences between our implementation and the original paper are addressed in the [Material Difference](https://github.com/JacksonCampolattaro/dgcnn-replication/tree/main?tab=readme-ov-file#material-differences) section of the code repository.

## Training hyperparameters:

The following table contains the hyperparameters we used for the training of the network on the ModelNet dataset:

|          Parameter | Setting          |
|-------------------:|:-----------------|
|         Batch size | 64               |
|          Optimizer | AdamW            |
| Base Learning Rate | 1E-3             |
|       Lr-scheduler | Cosine Annealing |
|    # of Points (N) | 1024             |
| # of Neighbors (k) | 20               |
|        # of Epochs | 250              |
|   Train/Test split | 80/20            |

## Classification results

In this subsection, we assess both the computational performance and the accuracy of the re-implemented DGCNN algorithm in the classification task.

### Computational performance

For testing the efficiency of the re-implemented DGCNN algorithm, we measure the running time in seconds per epoch.
The training is performed on a machine equipped with one Nvidia GeForce RTX 4090 GPU. 
The following table presents the results for the two implementations of the k-NN: torch-cluster and KeOps.

| KNN Implementation | Time per Epoch (s) |
|-------------------:|:------------------:|
|      Torch-cluster |       15.16        |
|              KeOps |       12.76        |

The results show that we can speed up the k-NN computation by replacing the current implementations such as torch-cluster with faster alternatives such as KeOps. However, KeOps has large performance benefits for 3d K-NN search, and this advantage fades out in higher dimensions, such as those found in the later layers of DGCNN.
Because the KNN search accounts for >45% of runtime, this can still produce a useful reduction in runtime per epoch.
### Accuracy

The provided table illustrates the reproducibility results for our re-implementation of the DGCNN algorithm. "Overall accuracy" represents the percentage of correctly classified points across the entire point cloud dataset, while "mean class accuracy" is the mean accuracy computed by averaging accuracy values for each class.

| DGCNN Implementation | Mean Class Accuracy | Overall Accuracy |
|---------------------:|:------------------:|------------------|
| Original paper | 90.2% | 92.9% |
| Ours | 88.3% | 91.9% |

Although our accuracies closely follow the original paper results, minor accuracy drops can be attributed to differences in implementation details, including varying augmentations (e.g., randomly dropping points), distinct training hyperparameters (e.g., using AdamW optimizer, different batch sizes), and slight changes in network architecture such as different implementations of batch normalization or the classification head. Nonetheless, our re-implementation demonstrates successful matching with the original paper, achieving accuracy levels close to the reported ones without doing the hyperparameter tuning.

# References

\[1\] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, **["Dynamic Graph CNN for Learning on Point Clouds,"](https://arxiv.org/abs/1801.07829)** ACM Trans. Graph., vol. 38, no. 5, Oct. 2019, Art. no. 146, pp. 1-12.

\[2\] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, and X. Tang and J. Xiao, **["3D ShapeNets: A Deep Representation for Volumetric Shapes,"](http://3dvision.princeton.edu/projects/2014/3DShapeNets/paper.pdf)** CVPR 2015.

\[3\] X. Zang, Y. Xie, S. Liao, J. Chen, and B. Yuan, **["Noise injection-based regularization for point cloud processing,"](https://arxiv.org/abs/2103.15027)** arXiv preprint arXiv:2103.15027, 2021.

\[4\] B. Charlier, J. Feydy, J. A. Glaunès, F.-D. Collin, and G. Durif, **["Kernel Operations on the GPU, with Autodiff, without Memory Overflows,"](https://arxiv.org/abs/2004.11127)** Journal of Machine Learning Research, vol. 22, no. 74, pp. 1-6, 2021.

