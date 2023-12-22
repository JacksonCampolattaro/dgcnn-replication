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
