---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 120

# Basic metadata
title: "Activation Functions"
date: 2020-08-17
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Nerual Network Basics"]
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
        parent: nn-basics
        weight: 2

---

Activation functions should be 

- **non-linear**
- **differentiable** (since training with Backpropagation)

## Sigmoid

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2011.07.47.png" alt="截屏2020-08-17 11.07.47" style="zoom:50%;" />
$$
\sigma(x)=\frac{1}{1+\exp (-x)}
$$

- Squashes numbers to range $[0,1]$

- ✅ <span style="color:green">Historically popular since they have nice interpretation as a saturating “firing rate” of a neuron</span>

- ⛔️ <span style="color:red">Problems</span>

  - **Vanishing gradients**: functions gradient at either tail of $1$ or $0$ is almost zero
  - Sigmoid outputs are **not zero-centered** (important for initialization)
  - $\exp()$ is a bit compute expensive

- Derivative
  $$
  \frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x))
  $$
  (See: [Derivative of Sigmoid Function](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x))

- Python implementation

  ```python
  def sigmoid(x):
      return 1 / (1 + np.exp(-x))
  ```

  - Derivative

    ```python
    def dsigmoid(y):
        return y * (1 - y)
    ```

## Tanh

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2011.08.03.png" alt="截屏2020-08-17 11.08.03" style="zoom:50%;" />

- Squashes numbers to range $[-1,1]$

- ✅ <span style="color:green">zero centered (nice)</span> :clap:
- ⛔️ <span style="color:red">**Vanishing gradients**: still kills gradients when saturated</span>

- Derivative:
  $$
  \frac{d}{dx}\tanh(x) = 1 - [\tanh(x)]^2
  $$

  ```python
  def dtanh(y):
      return 1 - y * y
  ```

## Rectified Linear Unit (ReLU)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2011.08.14.png" alt="截屏2020-08-17 11.08.14" style="zoom: 67%;" />
$$
f(x) = \max(0, x)
$$

- ✅ <span style="color:green">Advantages</span>
  - Does not saturate (in $[0, \infty]$)
  - Very computationally efficient
  - Converges much faster than sigmoid/tanh in practice

- ⛔️ <span style="color:red">Problems</span>

  - Not zero-centred output
  - No gradient for $x < 0$ (dying ReLU)

- Python implementation

  ```python
  import numpy as np
  
  def ReLU(x):
  	return np.maximum(0, x)
  ```

  

## Leaky ReLU

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2011.40.32.png" alt="截屏2020-08-17 11.40.32" style="zoom:50%;" />
$$
f(x) = \max(0.1x, x)
$$

- **Parametric Rectifier (PReLu)**
  $$
  f(x) = \max(\alpha x, x)
  $$

  - Also learn $\alpha$

- ✅ <span style="color:green">Advantages</span>

  - Does not saturate
  - Computationally efficient
  - Converges much faster than sigmoid/tanh in practice!
  - will not “die”

- Python implementation

  ```python
  import numpy as np
  
  def ReLU(x):
  	return np.maximum(0.1 * x, x)
  ```

## Exponential Linear Units (ELU)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17 11.08.41.png" alt="截屏2020-08-17 11.08.41" style="zoom: 50%;" />
$$
f(x) = \begin{cases} x &\text{if }x > 0 \\\\
\alpha(\exp (x)-1) & \text {if }x \leq 0\end{cases}
$$

- ✅ <span style="color:green">Advantages</span>
  - All benefits of ReLU
  - Closer to zero mean outputs
  - Negative saturation regime compared with Leaky ReLU (adds some robustness to noise)
- ⛔️ <span style="color:red">Problems</span>
  - Computation requires $\exp()$

## Maxout

$$
f(x) = \max \left(w\_{1}^{T} x+b\_{1}, w\_{2}^{T} x+b\_{2}\right)
$$

- Generalizes ReLU and Leaky ReLU
  - ReLU is Maxout with $w\_1 =0$ and $b\_1 = 0$
- ✅ <span style="color:green">Fixes the dying ReLU problem</span>
- ⛔️ <span style="color:red">Doubles the number of parameters</span>

## Softmax

- Softmax: probability that feature $x$ belongs to class $c\_k$
  $$
  o\_k = \theta\_k^Tx \qquad \forall k = 1, \dots, j
  $$

$$
p\left(y=c\_{k} \mid x ; \boldsymbol{\theta}\right)= p\left(c\_{k} = 1 \mid x ; \boldsymbol{\theta}\right) = \frac{e^{o\_k}}{\sum\_{j} e^{o\_j}}
$$

- Derivative:
  $$
  \frac{\partial p(\hat{\mathbf{y}})}{\partial o\_{j}} =y\_{j}-p\left(\hat{y}\_{j}\right)
  $$

## Advice in Practice

- Use <span style="color:green">**ReLU**</span>
  - Be careful with your learning rates / initialization
- Try out <span style="color:green">Leaky ReLU / ELU / Maxout</span>
- Try out <span style="color:orange">tanh</span> but don’t expect much
- <span style="color:red">Don’t use sigmoid</span>

## Summary and Overview

See: [Wiki-Activation Function](https://en.wikipedia.org/wiki/Activation_function)

