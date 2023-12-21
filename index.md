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


# Method:

In this section, we want to explain the working mechanism of the DGCNN method briefly. The distinguishing feature of the DGCNN method, setting it apart from previous approaches like PointNet, is that, instead of operating on individual points and aggregating their feature vectors, DGCNN generates an underlying graph on the points and performs convolution operations on features derived from the edges connecting neighboring points. Each layer is thus named "EdgeConv" to signify the convolution of edge features. Another difference from earlier methods is the dynamic nature of the graph in DGCNN. Unlike fixed graphs in previous models, the graph changes at each network layer. This dynamic graph is achieved by computing the k-nearest neighbors on the embeddings of the points (nodes of the graph) at each layer. In the following subsections, we explain edge convolution and dynamic graph updates.

## Edge convolution

EdgeConv takes as input a set of points and their corresponding features and computes the output features for each point. 
Each input point is represented by features associated with that point (such as its spatial coordinates, color, or intensity). 
Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be a directed graph, where $`\mathcal{V}`$ is the set of graph nodes with the size of $\lvert \mathcal{V}\rvert = N$, and $\mathcal{E}\subseteq\mathcal{V}\times\mathcal{V}$ is the set of graph edges. Suppose, we have $N$ points which are represented by the nodes of the graph $\mathcal{G}$, and node input and output features are denoted by sets of vectors 
$\{\mathbf{x}_i^{(in)}\}_{i=1}^{N}\subseteq \mathbb{R}^{F_{in}}$ $\{x_i^{(in)}\}{}$ $\\{$
and 
$`\{\mathbf{x}_i^{(out)}\}_{i=1}^{N}\subseteq \mathbb{R}^{F_{out}}`$, respectively. For each edge $(i,j)\in\mathcal{E}$, the edge features are defined as $\mathbf{e}_{ij}=h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})$, where function $h_{\mathbf{\Theta}}:\mathbb{R}^{F_{in}}\times\mathbb{R}^{F_{in}}\rightarrow \mathbb{R}^{F_{out}}$ maps the features of two neighboring nodes to the connecting edge features. The output features of node $i$ in EdgeConv layer can be obtained as
$$\mathbf{x}^{(out)}_i =\mathop{\square}_{j\in \mathcal{N}_i} h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)}),$$
where the operator $\square$ defines an aggregation operation that is permutation invariant. Examples of such operators are *summation* and *maximization*. The set $\mathcal{N}_i\subseteq\mathcal{V}$ define the neighboring nodes for the node $i$. 

For this paper, the authors choose the function $h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})$ as follows:

$$h_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)})={\bar h}_{\mathbf{\Theta}}(\mathbf{x}_i^{(in)},\mathbf{x}_j^{(in)} - \mathbf{x}_i^{(in)}),$$

where ${\bar h}_{\mathbf{\Theta}}$ is a multilayer perceptron network (MLP) with an input channel size of $2\,F_{in}$  and output channel size of $F_{out}$. The input to the MLP is the concatenated vector of $\mathop{concat}\big({\mathbf{x}_i^{(in)}},\mathbf{x}_j^{(in)} - \mathbf{x}_i^{(in)}\big)\in\mathbb{R}^{2\,F_{in}}$. The authors claim that with using 

$$i$$ this is inline.
