---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 150

# Basic metadata
title: "Backward Propagation"
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
        weight: 5

---

## Multi-Layer Perceptron (MLP)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2017.57.47.png" alt="截屏2020-08-17 17.57.47" style="zoom: 50%;" />

- **Input layer $I \in R^{D\_{I} \times N}$**

  - How we initially represent the features
  - Mini-batch processing with $N$ inputs

- **Weight matrices**

  - Input to Hidden: $W\_{H} \in R^{D\_{I} \times D\_{H}}$
  - Hidden to Output: $W\_{O} \in R^{D\_{H} \times D\_{O}}$

- **Hidden layer(s) $H \in R^{D\_{H} \times N}$**
  $$
  H = W\_{H}I + b\_H
  $$

  $$
  \widehat{H}\_j = f(H\_j)
  $$

  - $f$: non-linear activation function

- **Output layer $O \in R^{D_{O} \times N}$**

  - The value of the target function that the network approximates
    $$
    O = W\_O \widehat{H} + b\_O
    $$

- Loss function
  $$
  L = \frac{1}{2}(O - Y)^2
  $$

  - Achieve minimum $L$: **Stochastic Gradient Descent**

    1. Calculate $\frac{\delta L}{\delta w}$ for each parameter $w$

    2. Update the parameters
       $$
       w:=w-\alpha \frac{\delta L}{\delta w}
       $$

- Compute the gradients: **Backpropagation (Backprop)**

  - Output layer

  $$
  \begin{aligned}
  \frac{\delta L}{\delta O} &=(O-Y) \\\\ \\\\
  \frac{\delta L}{\delta \widehat{H}} &=W\_{0}^{T} \color{red}{\frac{\delta L}{\delta O}} \\\\ \\\\
  \frac{\delta L}{\delta W\_{O}} &= {\color{red}{\frac{\delta L}{\delta O} }}\widehat{H}^{T} \\\\ \\\\
  \frac{\delta L}{\delta b\_{O}} &= \color{red}{\frac{\delta L}{\delta O}}
  \end{aligned}
  $$

  ​	(<span style="color:red">Red</span>: terms previously computed)
  - Hidden layer (assuming $\widehat{H}=\operatorname{sigmoid}(H)$ )
    $$
    \frac{\delta L}{\delta H}={\color{red}{\frac{\delta L}{\delta \hat{H}}}} \odot \widehat{H} \odot(1-\widehat{H})
    $$

  - Input layer

  $$
  \begin{array}{l}
  \frac{\delta L}{\delta W\_{H}}={\color{red}{\frac{\delta L}{\delta H}}} I^{T} \\\\
  \frac{\delta L}{\delta b\_{H}}={\color{red}{\frac{\delta L}{\delta H}}}
  \end{array}
  $$

