---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 330

# Basic metadata
title: "Bolzmann Machine"
date: 2020-08-18
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Unsupervised Learning"]
categories: ["Deep Learning"]
toc: true # Show table of contents?

# Advanced metadata
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?

share: false  # Show social sharing links?
featured: true

comments: false  # Show comments?
disable_comment: true
commentable: false  # Allow visitors to comment? Supported by the Page, Post, and Docs content types.
editable: false  # Allow visitors to edit the page? Supported by the Page, Post, and Docs content types.

# Optional header image (relative to `static/img/` folder).
header:
  caption: ""
  image: ""

# Menu
menu: 
    deep-learning:
        parent: unsupervised-learning
        weight: 3

---

## **Boltzmann Machine**

- Stochastic recurrent neural network 
- Introduced by Hinton and Sejnowski 
- Learn internal representations 
- <span style="color:red">Problem: unconstrained connectivity</span>

### Representation

Model can be represented by Graph:

- Undirected graph

- Nodes: [States](states)

- [Edges: Dependencies between states](#connections)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2022.51.29.png" alt="截屏2020-08-18 22.51.29" style="zoom:50%;" />

### States

Types:

- **Visible states**
  - Represent observed data 
  - Can be input/output data
- **Hidden states**
  - Latent variable we want to learn
- **Bias states**
  - Always one to encode the bias

Binary states

- unit value $\in \\{0, 1\\}$

Stochastic

- Decision of whether state is active or not is stochastically

- Depend on the input
  $$
  z\_{i}=b\_{i}+\sum\_{j} s\_{j} w\_{i j}
  $$

  - $b\_i$: Bias
  - $S\_j$: State $j$
  - $w\_{ij}$: Weight between state $j$ and state $i$

  $$
  p\left(s\_{i}=1\right)=\frac{1}{1+e^{-z\_{i}}}
  $$

### Connections

- Graph can be fully connected (no restrictions)

- Unidircted:
  $$
  w\_{ij} = w\_{ji}
  $$

- No self connections:
  $$
  w\_{ii} = 0
  $$

### Energy

Energy of the network
$$
\begin{aligned}
E &= -S^TWS - b^TS \\\\
&= -\sum\_{i<j} w\_{i j} S\_{i} S\_{j}-\sum\_{i} b\_{i} s\_{i}
\end{aligned}
$$
Probability of input vector $v$
$$
p(v)= \frac{e^{-E(v)}}{\displaystyle \sum\_{u} e^{-E(u)}}
$$
Updating the nodes

- decrease the Energy of the network in average

- reach Local Minimum (Equilibrium)

- Stochastic process will avoid local minima
  $$
  \begin{array}{c}
  p\left(s\_{i}=1\right)=\frac{1}{1+e^{-z\_{i}}} \\\\
  z\_{i}=\Delta E\_{i}=E\_{i=0}-E\_{i=1}
  \end{array}
  $$

### Simulated Annealing

![截屏2020-08-18 23.53.50](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2023.53.50.png)

Use Temperature to allow for more changes in the beginning

- Start with high temperature
- “**anneal**” by slowing lowering T

- Can escape from local minima :clap:

### Search Problem

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2023.54.58.png" alt="截屏2020-08-18 23.54.58" style="zoom:67%;" />

- Input is set and fixed (clamped)

- Annealing is done

- Answer is presented at the output

- Hidden units add extra representational power

### Learning problem

- Situations
  - Present data vectors to the network

- Problem
  - Learn weights that generate these data with high probability

- Approach
  - Perform small updates on the weights 
  - Each time perform search problem

### Pros & Cons

✅ Pros

- Boltzmann machine with enough hidden units can compute any function

⛔️ Cons

- Training is very slow and computational expensive :cry:



## **Restricted Boltzmann Machine**

{{% alert note %}}

See also: [Restricted Boltzman Machine]({{< relref "rbm.md" >}})

 {{% /alert %}}

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-18%2023.58.36.png" alt="截屏2020-08-18 23.58.36" style="zoom:67%;" />

- Boltzmann machine with restriction

- Graph must be **bipartite**

  - Set of visible units

  - Set of hidden units

- ✅ Advantage

  - No connection between hidden units 
  - Efficient training

### Energy

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/v2-ede70fdae3090088792aab8607b3c2db_720w.jpg" alt="img" style="zoom:67%;" />

Energy: 
$$
\begin{aligned}
E(v, h) 
&= -a^{\mathrm{T}} v-b^{\mathrm{T}} h-v^{\mathrm{T}} W h \\\\
&= -\sum\_{i} a\_{i} v\_{i}-\sum\_{j} b\_{j} h\_{j}-\sum_{i} \sum_{j} v_{i} w_{i j} h_{j} 
\end{aligned}
$$
Probability of hidden unit:
$$
p\left(h\_{j}=1 \mid V\right)=\sigma\left(b\_{j}+\sum\_{i=1}^{m} W\_{i j} v\_{i}\right)
$$
Probability of input vector:
$$
p\left(v\_{i} \mid H\right)=\sigma\left(a\_{i}+\sum\_{j=1}^{F} W\_{i j} h\_{j}\right)
$$

> $$
> \sigma(x)=\frac{1}{1+e^{-x}}
> $$

Free Energy:
$$
\begin{array}{l}
e^{-F(V)}=\sum\_{j=1}^{F} e^{-E(v, h)} \\\\
F(v)=-\sum\_{i=1}^{m} v\_{i} a\_{i}-\sum_{j=1}^{F} \log \left(1+e^{z_{j}}\right) \\\\
z_{j}=b\_{j}+\sum\_{i=1}^{m} W\_{i j} v\_{i}
\end{array}
$$