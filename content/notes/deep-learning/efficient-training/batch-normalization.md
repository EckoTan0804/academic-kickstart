---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 22

# Basic metadata
title: "Batch Normalization"
date: 2020-08-16
draft: false
type: docs # page type
authors: ["admin"]
tags: ["Deep Learning", "Optimization", "Efficient training"]
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
        parent: efficient-training
        weight: 2

---

## Motivation: Feature scaling

Make different features have the same scaling (normalizing the data)

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2015.33.11.png" alt="截屏2020-05-20 15.33.11" style="zoom:67%;" />

- $x_i^r$: the $i$-th feature of the $r$-th input sample/instance

In general, gradient descent converges **much faster** with feature scaling than without it.

Illustration:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2015.35.33.png" alt="截屏2020-05-20 15.35.33" style="zoom: 33%;" />

### In hidden layer

![截屏2020-05-20 15.38.08](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2015.38.08.png)

From the point of view of Layer 2, its input is $a^1$, which is the output of Layer 1. As feature scaling helps a lot in training (gradient descent will converge much faster), can we also apply feature scaling for $a^1$ and the other hidden layer's output (such as $a^2$)?



## Internal Covariate Shift

In [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), the author's definition is:

> We define Internal Covariate Shift as the change in the distribution of network activations due to the change in network parameters during training.

In neural networks, the output of the first layer feeds into the second layer, the output of the second layer feeds into the third, and so on. When the parameters of a layer change, so does the distribution of inputs to subsequent layers.

These shifts in input distributions can be problematic for neural networks, especially deep neural networks that could have a large number of layers.

A common solution is to use small learning rate, but the training would then be slower. 😢



## Batch Nomalization (BN)

💪 Aim: solve internal covariate shift

### Batch

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2015.55.17.png" alt="截屏2020-05-20 15.55.17" style="zoom: 40%;" />

### Batch normalization

Usually we apply BN on the input of the activation function (i.e., **before** activation function )

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/dropin.jpg" alt="How does Batch Normalization Help Optimization? – gradient science" style="zoom: 15%;" />

Take the first hidden layer as example:

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2016.14.48.png" alt="截屏2020-05-20 16.14.48" style="zoom: 33%;" />

- Compute mean

$$
\mu = \frac{1}{N}\sum_{i=1}^{N}z^i
$$

- Compute standard deviation

$$
\sigma  = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(z^i-\mu)^2}
$$

- Normalize $z^i$ (Division is **element-wise**)
  $$
  \tilde{z}^i = \frac{z^i - \mu}{\sigma}
  $$

  - Now $\tilde{z}^i$ has zero mean and unit variance.

    - Good for activation function which could saturate (such as sigmoid, tanh, etc.)

      |                      Sigmoid witout BN                       |                       Sigmoid with BN                        |
      | :----------------------------------------------------------: | :----------------------------------------------------------: |
      | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/0*0CJqMLXgnZo1VqhS.jpeg" alt="img"  /> | <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/0*tPSfbtV7ILH0IN-I.jpeg" alt="img" style="zoom:55%;" /> |

- Scale and shift
  $$
  \hat{z}^{i}=\gamma \odot \tilde{z}^{i}+\beta
  $$

  - In practice, restricting the activations of each layer to be strictly zero mean and unit variance can limit the expressive power of the network. 
    - E,g,, some activation function doesn't require the input to be zero mean and unit variance
  - Scaling and shifting allow the network to learn input-independent parameters $\gamma$ and $\beta$ that can convert the mean and variance to any value that the network desires

**Note**: 

- Ideally, $\mu$ and $\sigma$ should be computed using the whole training dataset
- But this is expensive and infeasible
  - The size of training dataset is **enormous**
  - When $W^1$ gets updated, the output of the hidden layer will change, we have to compute $\mu$ and $\sigma$ again

- In practice, we can apply BN on batch of data, instead of the whole training dataset

  - But the size of bach can not be too small

  - If we apply BN on a small batch, it is difficult to estimate the mean ($\mu$) and the standard deviation ($\sigma$) of the WHOLE training dataset

    $\rightarrow$ The performance of BN will be bad!

### BN in Testing

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2016.31.25.png" alt="截屏2020-05-20 16.31.25" style="zoom:80%;" />

Problem: We do NOT have **batch** at testing stage. How can we estimate $\mu$ and $\sigma$?

Ideal solution: Compute $\mu$ and $\sigma$ using the whole training set

- But it is difficult in pratice
  - Traing set too large
  - Training could be online training

**Practical solution: Compute the moving average of $\mu$ and $\sigma$ of the batches during training**

### 👍 Benefit of BN 

- Reduce training times, and make very deep net trainable

  - Less covariate shift, we can use larger learning rate 
  - less exploding/vanishing gradients
    - Especailly effective for sigmoid, tanh, etc.

- Learning is less affected by parameters initialization

  <img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-05-20%2016.49.22.png" alt="截屏2020-05-20 16.49.22" style="zoom:40%;" />

- Reduces the demand for regularization, helps preventing overfitting

## Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- 👍 [Batch Normalization](https://www.youtube.com/watch?v=BZh1ltr5Rkg) 

- 👍 [Why need batch normalization?](https://www.youtube.com/watch?v=-5hESl-Lj-4)

