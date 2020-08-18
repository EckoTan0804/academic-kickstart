---
# Title, summary, and position in the list
# linktitle: ""
summary: ""
weight: 130

# Basic metadata
title: "Loss Functions"
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
        weight: 3

---

- Quantifies what it means to have a “good” model
- Different types of loss functions for different tasks, such as:
  - Classification
  - Regression 
  - Metric Learning
  - Reinforcement Learning

## Classification

- Classification: Predicting a discrete class label

- **Negative log-likelihood loss (per sample $x$)**
  $$
  L(\boldsymbol{x}, y)=-\sum\_{j} y_{j} \log p\left(c\_{j} \mid \boldsymbol{x}\right)
  $$
  - Used in various multiclass classification methods for NN training

- **Hinge Loss**: used in Support Vector Machines (SVMs)
  $$
  L(x, y)=\sum\_{j} \max \left(0,1-x\_{i} y\_{i}\right)
  $$

## Regression

- Regression: Predicting a one or multiple continuous quantities $y_1, \dots, y\_n$

- Goal: Minimize the distance between the predicted value $\hat{y}\_j$ and true values $y_j$

- **L1-Loss (Mean Average Error)**
  $$
  L(\hat{y}, y)=\sum\_{j}\left(\hat{y}\_{j}-x\_{j}\right)
  $$

- **L2-Loss (Mean Square Error, MSE)** 
  $$
  L(\hat{y}, y)=\sum\_{j}\left(\hat{y}\_{j}-x\_{j}\right)^2
  $$
  

## Metric Learning / Similarity Learning

- A model for measuring the distance (or similarity) between objects

- **Triplet Loss**

  ![截屏2020-08-17 12.20.34](https://raw.githubusercontent.com/EckoTan0804/upic-repo/master/uPic/截屏2020-08-17%2012.20.34.png)
  $$
  \sum_{(a, p, n) \in T} \max \left\\{0, \alpha-\left\|\mathbf{x}\_{a}-\mathbf{x}\_{n}\right\|\_{2}^{2}+\left\|\mathbf{x}\_{a}-\mathbf{x}\_{p}\right\|\_{2}^{2}\right\\}
  $$

