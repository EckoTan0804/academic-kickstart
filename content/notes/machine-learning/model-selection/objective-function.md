---
# Basic info
title: "Objective Function"
date: 2020-07-06
draft: false
type: docs
authors: ["admin"]
tags: ["ML", "Model Selection"]
categories: ["Machine Learning"]
toc: true

# Advanced settings
profile: false  # Show author profile?

reading_time: true # Show estimated reading time?
summary: "Objective function overview"
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

menu: 
    machine-learning:
        parent: model-selection
        weight: 1
---

## How does the objective function look like?

Objective function:

$$
\operatorname{Obj}(\Theta)= \overbrace{L(\Theta)}^{\text {Training Loss}}  + \underbrace{\Omega(\Theta)}_{\text{Regularization}}
$$

- Training loss: measures how well the model fit on training data
  $$
  L=\sum_{i=1}^{n} l\left(y_{i}, g_{i}\right)
  $$

  - Square loss: 
    $$
    l(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2
    $$
  - Logistic loss: 
    $$
    l(y_i, \hat{y}_i) = y_i \log(1 + e^{-\hat{y}_i}) + (1 - y_i) \log(1 + e^{\hat{y}_i})
    $$

- Regularization: How complicated is the model?
    - $L_2$ norm (Ridge): $\omega(w) = \lambda \|w\|^2$
    - $L_1$ norm (Lasso): $\omega(w) = \lambda \|w\|$
    

|                         | Objective Function                                           | Linear model? | Loss     | Regularization |
| ----------------------- | ------------------------------------------------------------ | ------------- | -------- | -------------- |
| **Ridge** regression    | $\sum_{i=1}^{n}\left(y_{i}-w^{\top} x_{i}\right)^{2}+\lambda\|w\|^{2}$ | ✅             | square   | $L_2$          |
| **Lasso** regression    | $\sum_{i=1}^{n}\left(y_{i}-w^{\top} x_{i}\right)^{2}+\lambda\|w\|$ | ✅             | square   | $L_2$          |
| **Logistic** regression | $\sum_{i=1}^{n}\left[y_{i} \cdot \ln \left(1+e^{-w^{\top} x_{i}}\right)+\left(1-y_{i}\right) \cdot \ln \left(1+e^{w^{\top} x_{i}}\right)\right]+\lambda\|w\|^{2}$ | ✅             | logistic | $L_1$          |



## Why do we want to contain two component in the objective? 

- **Optimizing training loss encourages predictive models** 

    - *Fitting well in training data at least get you close to training data which is hopefully close to the underlying distribution* 

- **Optimizing regularization encourages simple models** 

    - *Simpler models tends to have smaller variance in future predictions, making prediction* *stable* 