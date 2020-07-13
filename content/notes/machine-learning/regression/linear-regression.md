---
# Basic info
title: "Linear Regression"
date: 2020-07-06
draft: false
type: docs # page type
authors: ["admin"]
tags: ["ML", "Regression"]
categories: ["Machine Learning"]
toc: true # Show table of contents?

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: ""
share: false  # Show social sharing links?
featured: true
lastmod: true

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
    machine-learning:
        parent: regression
        weight: 1
---

## Linear Regression Model

A linear model makes a prediction $\hat{y}_i$ by **simply computing a weighted sum of the input $\boldsymbol{x}_i$, plus a constant $w_0$ called the _bias_ term**:

### For single sample/instances

$$
\hat{y}_i = f \left( \boldsymbol{x} \right) = w_0 + \sum\_{j=1}^{D}w\_{j} x\_{i, j}
$$

In matrix-form:

$$
\hat{y}\_{i}=w_{0}+ \displaystyle \sum\_{j=1}^{D} w_{j} x_{i, j}=\tilde{\boldsymbol{x}}\_{i}^{T} \boldsymbol{w}\
$$

- $\tilde{\boldsymbol{x}}\_{i} = \left[\begin{array}{c}{1} \\\\ {x_{i}}\end{array}\right] = \left[\begin{array}{c} {1} \\\\ x\_{i, 1} \\\\ \vdots \\\\ {x\_{i, D}}\end{array}\right] \in \mathbb{R}^{D+1}$

- $\boldsymbol{w}=\left[\begin{array}{c}{w\_{0}} \\\\ {\vdots} \\\\ {w\_{D}}\end{array}\right] \in \mathbb{R}^{D+1}$

### On full dataset

$$
\hat{\boldsymbol{y}}=\left[\begin{array}{c}{\hat{y}\_{1}} \\\\{\vdots} \\\\ {\hat{y}\_{n}}\end{array}\right]=\left[\begin{array}{c}{\tilde{\boldsymbol{x}}\_{1}^{T} \boldsymbol{w}} \\\\ {\vdots} \\\\ {\tilde{\boldsymbol{x}}\_{n}^{T} \boldsymbol{w}}\end{array}\right] = \underbrace{\left[\begin{array}{cc}{1} & {\boldsymbol{x}\_{1}^{T}} \\\\ {\vdots} & {\vdots} \\\\ {1} & {\boldsymbol{x}\_{n}^{T}}\end{array}\right]}\_{=: \boldsymbol{X}} \boldsymbol{w} = \boldsymbol{X} \boldsymbol{w}
$$

- $\hat{\boldsymbol{y}}$: vector containing the output for each sample
- $\boldsymbol{X}$: data-matrix containing a vector of ones as the first column as bias

> $y=\underbrace{\begin{bmatrix}{\widehat y}\_1 \\\\ \vdots\\\\{\widehat y}\_n\end{bmatrix}}\_{\boldsymbol\in\mathbf ℝ^{n\times1}}=\begin{bmatrix}\widehat x\_1^Tw\\\\\vdots\\\\\widehat x\_n^Tw\end{bmatrix}=\begin{bmatrix}1\cdot w\_0+x\_{1,1}\cdot w\_1+\cdots+x\_{1,D}\cdot w\_D\\\\\vdots\\\\1\cdot w\_0+x_{n,1}\cdot w\_1+\cdots+x\_{n,D}\cdot w_D\end{bmatrix}=\underset{=\begin{bmatrix}1&x\_1^T\\\\\vdots&\vdots\\\\1&x_n^T\end{bmatrix}\\\\=:\boldsymbol X\in\mathbb{R}^{n\times(1+D)}}{\underbrace{\begin{bmatrix}1&x\_{1,1}&\cdots&x\_{1,D}\\\\\vdots&\vdots&\ddots&\vdots\\\\1&x\_{n,1}&\cdots&x\_{n,D}\end{bmatrix}}\cdot}\underbrace{\begin{bmatrix}w\_0\\\\w_1\\\\\vdots\\\\w\_D\end{bmatrix}}\_{=:\boldsymbol w\boldsymbol\in\mathbf ℝ^{\boldsymbol(\mathbf1\boldsymbol+\mathbf D\boldsymbol)\boldsymbol\times\mathbf1}}$

