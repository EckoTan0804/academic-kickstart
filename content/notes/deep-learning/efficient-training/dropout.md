---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 230

# Basic metadata
title: "Dropout"
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
        weight: 3

---

## Model Overfitting

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-23%2022.00.46.png" alt="截屏2020-08-23 22.00.46" style="zoom:50%;" />

In order to give more "capacity"  to capture different features, we give neural nets a lot of neurons. But this can cause overfitting.

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-23%2021.59.37.png" alt="截屏2020-08-23 21.59.37" style="zoom: 50%;" />

Reason: Co-adaptation

- Neurons become dependent on others
- Imagination: neuron $H\_i$ captures a particular feature $X$ which however, is very frequenly seen with some inputs.
  - If $H\_i$ receives bad inputs (partial of the combination), then there is a chance that the feature is ignored 🤪

**Solution: Dropout!** :muscle:

## Dropout

With dropout the layer inputs become more sparse, forcing the network weights to become more robust.

![截屏2020-08-23 22.06.16](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-23%2022.06.16.png)

### Training

- Given 

  - input $X \in \mathbb{R}^D$ 
  - weights $W$ 
  - survival rate $p$

- Sample mask $M \in \{0, 1\}^D$ with $M\_i \sim \operatorname{Bernoulli}(p)$

- Dropped input:
  $$
  \hat{X} = X \circ M
  $$

- Perform backward pass and mask the gradients:
  $$
  \frac{\delta L}{\delta X}=\frac{\delta L}{\delta \hat{X}} \circ M
  $$

### Evaluation/Testing/Inference

- ALL input neurons $X$ are presented WITHOUT masking

- Because each neuron appears with probability $p$ in training 

  $\to$ So we have to scale $X$ with $p$ (or scale $\hat{X}$ with $\frac{1}{1-p}$ during training) to match its expectation



## Why Dropout works?

- Each of the “dropped” instance is a different network configuration
- $2^n$ different networks sharing weights
- The inference process can be understood as an **ensemble of $2^n$ different configuration**
- This interpretation is in-line with *Bayesian Neural Networks*

<img src="https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-23%2022.20.36.png" alt="截屏2020-08-23 22.20.36" style="zoom: 50%;" />