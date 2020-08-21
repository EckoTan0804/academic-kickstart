---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 610

# Basic metadata
title: "Recurrent Neural Networks"
date: 2020-08-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "RNN"]
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
        parent: rnn
        weight: 1
---

{{% alert note %}}
For detailed explanation and summary see: [RNN Summary]({{< relref "../../natural-language-processing/RNN/rnn-summary.md" >}})
{{% /alert %}}

## Overview

- Specifically designed for long-range dependency
- 💡 Main idea: connecting the hidden states together within a layer

## Simple RNNs

### Elman Networks

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2000.09.01.png" alt="截屏2020-08-21 00.09.01" style="zoom:80%;" />

- The output of the hidden layer is used as input for the next time step
- They use a copy mechanism. Previous state of H copied into next iteration of net
  - Do NOT really have true recurrence, no backpropagation through time

### Jordan Nets

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2000.15.46.png" alt="截屏2020-08-21 00.15.46" style="zoom:80%;" />

- Same Copy Mechanisms as Elman Networks, but from Output Layer

## RNN Structure

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2000.19.34.png" alt="截屏2020-08-21 00.19.34" style="zoom: 50%;" />
$$
H\_{t}=f\_{a c t}\left(W^{H} H\_{t-1}+W^{X} X\_{t}+b\right)
$$

- $f\_{act}$: activation function (sigmoid, tanh, ReLU ... )
- Inside the bracket: a linear “voting” combination between the memory $H\_{t-1}$ and input $X\_t$



## Learning in RNNs: BPTT

### One-to-One RNN

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2009.30.48.png" alt="截屏2020-08-21 09.30.48" style="zoom: 50%;" />

- Identical to a MLP

- $W^H$ is non-existent or an identity function

- Backprop (obviously) like an MLP

  - $\frac{\delta L}{\delta O}$ is directly computed from $L(O, Y)$

  $$
  \begin{array}{l}
  \frac{\delta L}{\delta H\_{1}}=\frac{\delta L}{\delta O} \frac{\delta O}{\delta H\_{1}}=W^{O} \color{red}{\frac{\delta L}{\delta O}} \\\\
  \frac{\delta L}{\delta W^O}={\color{red}{\frac{\delta L}{\delta O}}} \frac{\delta L}{\delta W^{O}}={\color{red}{\frac{\delta L}{\delta O}}} H^{T}
  \end{array}
  $$

  ​	(<span style="color:red">Red terms</span> are precomputed values in backpropagation)

### Many-to-One RNN

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2009.58.38.png" alt="截屏2020-08-21 09.58.38" style="zoom:50%;" />

- Often used to sequence-level classification 
- For example:
  - Movie review sentiment analysis
  - Predicting blood type from DNA
- Input: sequence $I \in \mathbb{R}^{T \times D\_I}$

- Output: vector $O \in \mathbb{R}^{D\_O}$

- Backprop

  - The gradient signal is **backpropagated through time** by the recurrent connections
  - **Shared weights**: the weight matrices are duplicated over the time dimension
  - In each time step, the weights $\theta = (W^H, W^I, W^O)$ establishes a multi-layer
    perceptron.

  ![截屏2020-08-21 10.01.30](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2010.01.30.png)

  ![截屏2020-08-21 10.08.30](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2010.08.30.png)

  - The gradients are **summed up** over time
    $$
    \begin{aligned}
    \frac{\delta L}{\delta W^{I}} &=\frac{\delta L}{\delta W\_{1}^{I}}+\frac{\delta L}{\delta W_{2}^{I}}+\cdots+\frac{\delta L}{\delta W\_{T}^{I}} \\\\ \\\\
    \frac{\delta L}{\delta W^{H}} &=\frac{\delta L}{\delta W\_{1}^{H}}+\frac{\delta L}{\delta W\_{2}^{H}}+\cdots+\frac{\delta L}{\delta W\_{T-1}^{H}}
    \end{aligned}
    $$

### Many-to-Many RNN

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2010.14.35.png" alt="截屏2020-08-21 10.14.35" style="zoom:50%;" />

- The inputs and outputs have the **same** number of elements

- Example

  - Auto-regressive models
  - Character generation model
  - Pixel models

- Backprop

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2010.26.26.png" alt="截屏2020-08-21 10.26.26" style="zoom:50%;" />

  - Loss function
    $$
    L = L\_1 + L\_2 + \cdots + L\_T
    $$

  - $H\_t$ can not change $O\_{k<t}$, so we have
    $$
    \frac{\delta L}{\delta H\_t} = \sum\_{i \geq t}\frac{\delta L\_i}{\delta H\_t}
    $$
    (Because we cannot change the past so the terms with $i < t$ can be omitted)

    ![截屏2020-08-21 10.29.12](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2010.29.12.png)

  - After getting the $\frac{\delta L}{\delta H\_t}$ terms, we can compute the necessary gradients for $\frac{\delta L}{\delta W^{O}}, \frac{\delta L}{\delta W^{H}}$ and $\frac{\delta L}{\delta W^{I}}$

  - For shared weights $W^O$, apply the same principle in the Many-to-One RNN.

### One-to-Many RNN

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2011.46.58.png" alt="截屏2020-08-21 11.46.58" style="zoom:50%;" />

- More rarely seen than Many-to-One
- For example:
  - Image description generation
  - Music generation

- Input: vector $I \in \mathbb{R}^{D\_{I}}$

- Output: sequence $O \in R^{T \times D\_{O}}$

## Problem of BPTT

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2011.51.23.png" alt="截屏2020-08-21 11.51.23" style="zoom:50%;" />
$$
\begin{array}{l}
\frac{\delta L}{\delta H\_{114}}=\frac{\delta L}{\delta H\_{115}} \frac{\delta H\_{115}}{\delta H\_{114}} \\\\ \\\\
\frac{\delta L}{\delta H_{113}}=\frac{\delta L}{\delta H\_{115}} \frac{\delta H\_{115}}{\delta H\_{114}} \frac{\delta H\_{114}}{\delta H\_{113}}=W^{O} f^{\prime}\left(H\_{115}\right) W^{H} f^{\prime}\left(H\_{114}\right) W^{H}
\end{array}
$$
If we wand to send signals back to step 50:
$$
\begin{aligned}
\frac{\delta L}{\delta H\_{50}} &=\frac{\delta L}{\delta H\_{115}} \frac{\delta H\_{115}}{\delta H\_{114}} \ldots \frac{\delta H\_{51}}{\delta H\_{50}} \\\\ \\\\
&=W^{O} f^{\prime}\left(H\_{115}\right) W^{H} f^{\prime}\left(H\_{114}\right) W^{H} f^{\prime}\left(H\_{113}\right) W^{H} \ldots
\end{aligned}
$$
65 times of repetition for $f^{\prime}\left(H\_{115}\right) W^{H}$!!! 😱😱😱

Assuming network is univariate $W^H$ has one element $w$ and $f$ is either ReLU or linear

- If $w = 1.1$: $\frac{\delta L}{\delta H_{50}}=117$

- If $w = 0.9$: $\frac{\delta L}{\delta H_{50}}=0.005$

In general

- if the maximum eigenvalue of $W^H > 1$: $\frac{\delta L}{\delta H_{T}}$ is likely to <span style="color:red">explode</span>

- if the maximum eigenvalue of $W^H < 1$: $\frac{\delta L}{\delta H_{T}}$ is likely to <span style="color:red">vanish</span>

  ((small gradients make the RNN unable to learn the important features to solve the problem))

### Gradient clipping

- Simple solution for gradient exploding

- Clip the gradient $g = \frac{\delta L}{\delta w}$ so 
  $$
  \|g\|>\mu: g=\mu \frac{g}{\|g\|} \qquad \text{(biased gradients)}
  $$

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-21%2012.05.02.png" alt="截屏2020-08-21 12.05.02" style="zoom:50%;" />

## BPTT Example

{{< youtube RrB605Mbpic >}}